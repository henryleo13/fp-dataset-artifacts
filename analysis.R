# Read JSONs and evaluate


# load libraries
library(jsonlite)
library(tidyverse)
library(ggthemes)
library(patchwork)
library(rio)

# Type dir in console
dir <- "D:/OneDrive/MSDS/Natural_Language_Processing/fp-dataset-artifacts/"
load_metric_data <- function (file_path, file_name) {
  df <- import(file_path) |> 
    mutate(file_name = file_name)
}
read_eval_metrics <- function(eval_loc, eval_name) {
  df_return <- eval_loc |> 
    readLines() |> 
    map_df(fromJSON) |> 
    select(eval_loss, eval_accuracy)
  
  return(df_return)
}

#### Data set Analysis ####
eval_names <- c("dev_matched", "dev_mismatched", "hans", "mnli_train")
eval_locs <- c("dev_matched", "dev_mismatched", "HANS", "train") %>%
  str_c(dir, "evals/base3e/", ., "/eval_predictions.jsonl")

label_mapping <- tibble(
  label = c(0, 1, 2),
  label_name = c("entailment", "neutral", "contradiction")
)


# Read HANS Json
hans_file_name <- str_c(dir, "datasets/heuristics_evaluation_set.jsonl")
hans <- hans_file_name |> 
  readLines() |> 
  map(fromJSON) %>% 
  map(as_tibble, .progress = T) %>% 
  bind_rows() |> 
  select(heuristic, template, sentence1, sentence2, gold_label)

hans_eval <- eval_locs[3] |> 
  readLines() |> 
  map(fromJSON) %>% 
  map(as_tibble, .progress = T) %>% 
  bind_rows() |> 
  select(-predicted_scores, -label) |> 
  unique()

# Create template table
templates <- str_c("temp",seq(1,68))

template_description <- c(
  # Lexical Overlap
  "Simple", 
  rep("Preposition on subject", 6),
  rep("Relative Clause on Subject", 12),
  rep("Passive Incorrect", 2),
  rep("Conjuctions", 4),
  rep("Relative Clause", 4),
  "Across PP",
  "Across Relative Clause",
  rep("Conjuctions",2),
  "Across Adverb",
  rep("Passive, 2", 2),
  #Subsequence
  "NPS", 
  "PP on subject",
  "Relative Clause on Subject",
  rep("Past particles", 2),
  rep("NP/Z", 2),
  rep("Conjoined Subject", 2),
  "Modified Plural Subject",
  "Understood Argument",
  "Relative Clause",
  "PP",
  # Constituent
  "Outside Embedded Clause",
  "Outside If",
  "Said",
  rep("Disjunction",2),
  "Noun Complements",
  rep("Adjective Complements",2),
  "Probably, Supposedly",
  "Since",
  "Adverb Outside",
  "Knew",
  rep("Conjunction",2),
  "Embedded Question",
  "Noun Complement",
  rep("Adjective Complement", 2),
  "Sentential Adverb"
)

temp_table <- tibble(templates, template_description)


df_hans <- bind_cols(hans, hans_eval) |> 
  mutate(
    pred_label = case_when(
      predicted_label == 0 ~ "entailment",
      predicted_label == 1 ~ "non-entailment",
      T ~ NA_character_,
    ),
    heuristic = factor(heuristic, levels = c("lexical_overlap", "subsequence", "constituent")),
    correct = ifelse(gold_label == pred_label, 1, 0)
  ) |> 
  select(-predicted_label) |> 
  left_join(temp_table, by = c("template" = "templates"))

df_hans |> 
  group_by(heuristic, template_description, gold_label) |> 
  summarize(
    accuracy = mean(correct)
  ) |> 
  filter(gold_label == "non-entailment", !is.na(accuracy)) |> view()

df_hans |> 
  group_by(heuristic, gold_label) |> 
  summarize(
    accuracy = mean(correct)
  ) |> 
  ggplot(aes(x = heuristic, y = accuracy, fill = fct_rev(gold_label))) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(
    name = "Gold Label",
    labels = c("Entailment", "Non-entailment"),
    option = "E", # E, H
    direction = -1
  ) + 
  scale_y_continuous(
    expand = expansion(mult = c(0, .05)),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_discrete(
    name = "Heuristic",
    # Capitalize first letter
    labels = function(x) str_to_title(x)
  ) +
  # flip axis
  coord_flip() +
  cowplot::theme_minimal_vgrid()+
  theme(
    legend.position = "bottom",
  )

ggsave(str_c(dir, "plots/mnli_on_hans.png"), width = 6, height = 8, units = "in", dpi = 300)


read_evals <- function(eval_loc, eval_name) {
  df_return <- eval_loc |> 
    readLines() |> 
    map(fromJSON) %>% 
    map(as_tibble, .progress = T) %>% 
    bind_rows() |> 
    select(-predicted_scores) |> 
    unique() |> 
    left_join(label_mapping, by = c("predicted_label" = "label")) |>
    rename(pred_label = label_name) |> 
    left_join(label_mapping, by = c("label" = "label")) |>
    rename(gold_label = label_name) |> 
    mutate(
      gold_label2 = if_else(gold_label == "entailment", "entailment", "non-entailment"),
      pred_label2 = if_else(pred_label == "entailment", "entailment", "non-entailment"),
      correct = ifelse(label == predicted_label, 1, 0),
      correct2 = ifelse(gold_label2 == pred_label2, 1, 0)
    )

  df_return <- df_return |> 
    mutate(eval_name = eval_name)
  
  return(df_return)
}

df <- map2(eval_locs, eval_names, read_evals) |> 
  bind_rows() |> 
  mutate(
    eval_name = factor(eval_name, levels = c("mnli_train", "dev_matched", "dev_mismatched", "hans")),
    model = "base",
    train_epochs = 3
  )

p1 <- df |>
  group_by(eval_name) |> 
  summarize(
    accuracy = mean(correct)
  ) |>
  ggplot(aes(x = eval_name, y = accuracy)) +
  geom_col(fill = "skyblue") + 
  geom_text(aes(label = scales::percent(accuracy)), vjust = -0.5) +
  scale_y_continuous(
    limits = c(0, 1.05),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent
  ) +
  scale_x_discrete(
    name = NULL,
    # Capitalize first letter
    labels = function(x) str_to_upper(str_replace(x,"_", "\n"))
  ) +
  cowplot::theme_minimal_hgrid()+
  labs(tag = "(a)") + 
  theme(
    plot.tag.position = "bottom", 
    plot.tag = element_text(face = "plain")
  )
p1

p2 <- df |> 
  group_by(eval_name, gold_label2) |> 
  summarize(
    accuracy = mean(correct2),
    count = n()
  ) |> 
  filter(!is.na(gold_label2)) |> 
  ggplot(aes(x = eval_name, y = accuracy, fill = gold_label2)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = scales::percent(accuracy)), position = position_dodge(width = 0.9), vjust = -0.5) + 
  scale_fill_viridis_d(
    name = "Gold Label",
    option = "E"
    #labels = c("Entailment", "Non-entailment")
  ) +
  scale_y_continuous(
    limits = c(0, 1.05),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent
  ) +
  scale_x_discrete(
    name = NULL,
    # Capitalize first letter
    labels = function(x) str_to_upper(str_replace(x,"_", "\n")),
    expand = expansion(mult= c(0, 0.3))
  ) +
  cowplot::theme_minimal_hgrid()+labs(tag = "(b)") + 
  theme(
    plot.tag.position = "bottom", 
    plot.tag = element_text(face = "plain"),
    # move legend inside plot
    legend.position = c(0.83, 0.5),
  )
p2

# use patchwork, put p1 and p2 in one row with p1 a third of the width
p1 + p2 + plot_layout(widths = c(1, 2))
ggsave(str_c(dir, "plots/electra_base.png"), width = 15, height = 5, units = "in", dpi = 300)

#patchwork1 + plot_annotation(tag_levels = 'a', tag_prefix = "(", tag_suffix = ")")

get_overlap <- function(hyp_words, premise) {
  premise <- str_to_lower(premise)
  ratio <- sum(map_dbl(hyp_words, ~sum(str_detect(premise, .x)))) / length(unlist(hyp_words))
  
  if (length(ratio) == 1) {
    return(ratio)
  } else {
    return(NA)
  }
  
}
# Investigate incorrect on Dev matched and mismatched
df_overlap <- df |> 
  filter(eval_name %in% c("dev_matched", "dev_mismatched")) |> 
  filter(correct == 0) |> 
  select(premise, hypothesis, gold_label, pred_label, eval_name) |> 
  # calculate percentage of words in hypothesis that show up in premise
  mutate(
    hypothesis_words = str_split(hypothesis, " ", simplify = F) |> 
      map(str_to_lower),
    premise_length = str_length(premise)
  )

df_overlap <- df_overlap |> 
  #filter(premise_length < 50) |> 
  mutate(
    overlap = map2_dbl(hypothesis_words, premise, possibly(get_overlap, otherwise = NA_real_))
  ) 

df_overlap |> 
  export(str_c(dir, "evals/incorrect_dev.xlsx"))

# Create distribution of overlap by predicted label
df_overlap |> 
  filter(!is.na(overlap)) |> 
  ggplot(aes(x = overlap, fill = pred_label)) +
  geom_density(alpha = 0.5) +
  #facet_wrap(~eval_name, ncol = 1) +
  scale_fill_viridis_d(
    name = "Predicted Label",
    option = "E"
  ) +
  scale_x_continuous(
    name = "Overlap",
    breaks = seq(0, 1, 0.1),
    limits = c(0, 1)
  ) +
  scale_y_continuous(
    name = "Count"
  ) +
  cowplot::theme_minimal_hgrid() 

#### Biased Model on Synthetic Data ####
file_with_path <- list.files(path = str_c(dir, "synthetic_data/"), recursive = T, pattern = ".csv", full.names = T)
file_names <- list.files(path = str_c(dir, "synthetic_data/"), recursive = T, pattern = ".csv") |> 
  stringr::word(1, sep = "/") 
df_syn_metrics <- map2(file_with_path, file_names, load_metric_data) |> 
  bind_rows() |> 
  rename(
    "dev_acc" = "acc",
    "dev_c_above_90" = "c_above_90"
  ) |> 
  select(-contains("loss")) |> 
  mutate(
    `training samples` = str_extract(file_name, "[0-9]+") |> as.numeric(),
    `training samples text` = factor(str_c(`training samples`, "k"))|> fct_reorder(`training samples`)
  ) |> 
  # Replace first occurence of _ with / in col names
  rename_with(~str_replace(., "_", "/"), contains("_")) |>
  pivot_longer(
    cols = contains(c("c_above_90", "acc")),
    names_to = c("eval_set", "metric"),
    names_pattern = "(.*)/(.*)",
    values_to = "value"
  ) |> 
  mutate(
    value = if_else(value < 0, NA_real_, value)
  ) |> 
  pivot_wider(
    names_from = "metric",
    values_from = "value"
  ) 
  
df_start_values <- df_syn_metrics |> 
  select(`file/name`, `training samples`, `training samples text`, eval_set) |> 
  unique() |> 
  mutate(epoch = 0, acc = 0.33, c_above_90 = 0)

psyn1 <- df_syn_metrics |>
  filter(!is.na(acc)) |> 
  bind_rows(df_start_values) |> 
  ggplot(aes(x = epoch, y = acc, color = eval_set)) +
  #ignore na when drawing line
  geom_line() +
  geom_line(aes(y = c_above_90), linetype = "dashed") +
  geom_point() +
  facet_wrap(~`training samples text`, nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  )  +
  scale_color_viridis_d(
    #palette = "Set1",
    option = "C", #C
    name = "Evaluation Set",
    labels = c("MNLI-Antibias", "MNLI-Bias", "MNLI")
  ) +
  scale_x_continuous(
    name = "Epoch",
    breaks = seq(1, 10, 2),
    expand = expansion(mult = 0)
  ) +
  labs(
    tag = "(a)"
  ) +
  theme_bw(16)+
  theme(
    # Don't show legend
    legend.position = "none",
    plot.tag.position = "bottom", 
    plot.tag = element_text(face = "plain")
  ) 
psyn1
  
  
# zoom in on 10k
psyn2 <- df_syn_metrics |>
  bind_rows(df_start_values) |>
  pivot_longer(
    cols = contains(c("c_above_90", "acc")),
    names_to = "metric",
    values_to = "value"
  ) |> 
  filter(!is.na(value), `training samples text` == "10k", epoch >=0, epoch <=3) |> 
  ggplot(aes(x = epoch, y = value, color = eval_set)) +
  #facet_wrap(~`training samples text`, nrow = 1)+
  #ignore na when drawing line
  geom_line(aes(linetype = metric)) +
  #geom_line(aes(y = c_above_90), linetype = "dashed") +
  geom_point() +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  )  +
  scale_color_viridis_d(
    #palette = "Set1",
    option = "C", #C
    name = "Evaluation Set",
    labels = c("MNLI-Antibias", "MNLI-Bias", "MNLI")
  ) +
  scale_x_continuous(
    name = "Epoch",
    breaks = seq(1, 10, 1),
    expand = expansion(mult = 0)
  ) +
  labs(tag = "(b)")+
  theme_bw(16) +
  # Add legend for solid line vs dashed line
  scale_linetype_manual(
    name = "Metric",
    values = c("solid", "dashed"),
    labels = c("Accuracy", "Prediction % \nabove 90% Confidence")
  ) +
  theme(
    #legend.position = c(0.7, 0.85),
    plot.tag.position = "bottom", 
    plot.tag = element_text(face = "plain")
  )
psyn2
# use patchwork, put p1 and p2 in one row with p1 a third of the width
psyn1 + psyn2 + plot_layout(widths = c(4, 1))
ggsave(str_c(dir, "plots/synthetic.png"), width = 15, height = 5, units = "in", dpi = 300)


  #### Biased Model Performance ####
file_with_path <- list.files(path = str_c(dir, "plotting_data"), pattern = ".csv", full.names = T)
file_names <- list.files(path = str_c(dir, "plotting_data"), pattern = ".csv")

load_metric_data <- function (file_path, file_name) {
  df <- import(file_path) |> 
    mutate(file_name = file_name)
}
df_eval_metrics <- map2(file_with_path, file_names, load_metric_data) |> 
  bind_rows()

df_eval_metrics |> 
  mutate(
    training_samples = str_extract(file_name, "[0-9]+") |> as.numeric(),
    training_samples_text = factor(str_c(training_samples, "k"))|> fct_reorder(training_samples)
  ) |> 
  ggplot(aes(x= epoch, y = acc, color = training_samples_text)) +
  geom_line(size = 1) +
  scale_color_brewer(
    palette = "Set1",
    name = "Training Samples",
    #labels = c("5k", "10k", "15k", "20k")
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_continuous(
    limits = c(0, 10),
    expand = expansion(mult = 0),
    name = "Epoch",
    breaks = seq(0, 10, 1)
  ) +
  cowplot::theme_minimal_grid()

df_eval_metrics |> 
  mutate(
    training_samples = str_extract(file_name, "[0-9]+") |> as.numeric(),
    training_samples_text = factor(str_c(training_samples, "k")) |> fct_reorder(training_samples)
  ) |> 
  ggplot(aes(x= epoch, y = c_above_90, color = training_samples_text)) +
  geom_line(size = 1) +
  scale_color_colorblind(
    name = "Training Samples",
    #labels = c("5k", "10k", "15k", "20k")
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "c above 90",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_continuous(
    limits = c(0, 10),
    expand = expansion(mult = 0),
    name = "Epoch",
    breaks = seq(0, 10, 1)
  ) +
  cowplot::theme_minimal_grid()

#### Debiased vs Base ####
debiased_run_path <- str_c(
  dir, 
  "/student_model/student_model_5e_bias10k2e/model/runs/",
  "Dec03_14-27-46_d32e6d74b184/",
  "events_out_tfevents_1701613784_d32e6d74b184_5200.csv"
)

base_run_path <- str_c(
  dir, 
  "/trained_model/mnli10/runs/",
  "Nov27_21-07-11_86ab0c42b32c/",
  "events_out_tfevents_1701119310_86ab0c42b32c_126365.csv"
)

file_with_path <- c(base_run_path, debiased_run_path)
file_names <- c("base", "debiased")
df_eval_metrics <- map2(file_with_path, file_names, load_metric_data) |> 
  bind_rows()

# plot of t_loss vs epoch by file_name
df_eval_metrics |> 
  ggplot(aes(x= epoch, y = t_loss, color = file_name)) +
  geom_line(size = 1) +
  scale_color_viridis_d(
    option = "E",
    name = "Model",
    labels = c("Base", "Debiased")
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Training Loss",
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_continuous(
    limits = c(0, 10),
    expand = expansion(mult = 0),
    name = "Epoch",
    breaks = seq(0, 10, 1)
  ) +
  cowplot::theme_minimal_grid()

eval_names_debiased <- c("dev_matched", "dev_mismatched", "hans")
eval_locs_debiased_3e <- c("dev_matched", "dev_mismatched", "HANS") %>%
  str_c(dir, "evals/student_model_3e_bias10k2e/", ., "/eval_predictions.jsonl")
eval_locs_debiased_10e <- c("dev_matched", "dev_mismatched", "HANS") %>%
  str_c(dir, "evals/student_model_10e_bias10k2e/", ., "/eval_predictions.jsonl")


df_debiased_3e <- map2(eval_locs_debiased_3e, eval_names_debiased, read_evals) |> 
  bind_rows() |> 
  mutate(
    eval_name = factor(eval_name, levels = c("dev_matched", "dev_mismatched", "hans")),
    model = "Debiased 3 epochs",
    train_epochs = 3
  )

df_debiased_10e <- map2(eval_locs_debiased_10e, eval_names_debiased, read_evals) |> 
  bind_rows() |> 
  mutate(
    eval_name = factor(eval_name, levels = c("dev_matched", "dev_mismatched", "hans")),
    model = "Debiased 10 epochs",
    train_epochs = 10
  )

df |> 
  bind_rows(df_debiased_3e, df_debiased_10e) |> 
  group_by(model, eval_name, train_epochs) |>
  summarise(
    accuracy = mean(correct)
  ) |>
  filter(eval_name != "mnli_train") |> 
  ggplot(aes(x = eval_name, y = accuracy, fill = model)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(
    option = "E",
    name = "Model",
    #labels = c("Base", "Debiased 3 epochs", "Debiased 10 epochs")
  )+
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_discrete(
    name = "Evaluation Set",
    labels = c("dev_matched", "dev_mismatched", "HANS")
  ) +
  facet_wrap(~train_epochs) +
  cowplot::theme_minimal_grid()


#### Annealing ####
eval_annealed_theta <- c(50, 60, 70, 80, 90)

eval_names_debiased <- c("dev_matched", "dev_mismatched", "HANS")
eval_locs_annealed <- expand.grid(
  shallow_model = c("15K", "10k2e"),
  theta = c("a50", "a60", "a70", "a80", "a90", "a100"),
  eval_name = c("dev_matched", "dev_mismatched", "HANS")
) |> 
  mutate(
    metrics_loc = str_c(dir,"evals/annealed/", shallow_model, "/", theta, "/", eval_name, "/eval_metrics.json")
  ) 
  
empty_tibble = tibble(eval_loss = NA, eval_accuracy = NA)

annealed_metrics <- map(
  eval_locs_annealed$metrics_loc, 
  possibly(read_eval_metrics, otherwise = empty_tibble)
) |> 
  bind_rows() |> 
  bind_cols(eval_locs_annealed)

annealed_metrics |>
  filter(shallow_model == "10k2e") |> 
  ggplot(aes(x = theta, y = eval_accuracy, fill = eval_name)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(
    option = "E",
    name = "Model",
    #labels = c("15K", "10k2e")
  )+
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_discrete(
    name = "Annealing Theta",
    labels = c("50", "60", "70", "80", "90", "control")
  ) +
  #facet_wrap(~eval_name, ncol = 1) +
  cowplot::theme_minimal_grid()


#### Combined Data Analysis ####
eval_names_debiased <- c("dev_matched", "dev_mismatched", "HANS")

biased_models <- tibble(
  model =c("base", "debiased", "debiased"),
  shallow_model = c("base", "10k2e", "15k10e")
) 

annealed_models <- tibble(
  epoch = rep("3e",6),
  model = rep("debiased", 6),
  min_theta_text = c("a50", "a60", "a70", "a80", "a90", "a100"),
  min_theta = seq(50, 100, by = 10)
)

empty_tibble = tibble(eval_loss = NA, eval_accuracy = NA)

eval_locs <- expand.grid(
  model = c("debiased", "base"),
  eval_name = c("dev_matched", "dev_mismatched", "HANS"),
  epoch = c("3e", "10e")
) |> 
  left_join(biased_models, by = "model") |> 
  left_join(annealed_models, by = c("epoch" = "epoch", "model" = "model")) |> 
  mutate(
    min_theta_text = if_else(is.na(min_theta_text), "a100", min_theta_text),
    min_theta = if_else(is.na(min_theta), 100, min_theta),
    metrics_loc = str_c(dir,"evals/", model, "/", shallow_model, "/", epoch, "/", min_theta_text, "/", eval_name, "/eval_metrics.json")
  ) 

eval_metrics <- map(
  eval_locs$metrics_loc, 
  possibly(read_eval_metrics, otherwise = empty_tibble)
) |> 
  bind_rows() |> 
  bind_cols(eval_locs)


eval_metrics |> 
  filter(min_theta_text == "a100") |> 
  mutate(
    shallow_model = factor(shallow_model, levels = c("base", "10k2e", "15k10e")),
    epoch = factor(epoch, levels = c("3e", "10e")),
  ) |> 
  kable(format = "latex") |> 
  kableExtra::kable_classic()


eval_metrics |> 
  filter(min_theta_text == "a100") |> 
  mutate(
    shallow_model = factor(shallow_model, levels = c("base", "10k2e", "15k10e")),
    epoch = factor(epoch, levels = c("3e", "10e")),
  ) |> 
  ggplot(aes(x = eval_name, y = eval_accuracy, fill = shallow_model)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(
    option = "E",
    name = "Model",
    #labels = c("Base", "Debiased 3 epochs", "Debiased 10 epochs")
  )+
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_discrete(
    name = "Evaluation Set",
    #labels = c("dev_matched", "dev_mismatched", "HANS")
  ) +
  facet_wrap(~epoch) +
  cowplot::theme_minimal_grid()

eval_metrics |> 
  filter(
    epoch == "3e",
    shallow_model != "base"
  ) |> 
  mutate(
    shallow_model = factor(shallow_model, levels = c("10k2e", "15k10e")),
    min_theta_text = factor(min_theta_text, levels = c("a50", "a60", "a70", "a80", "a90", "a100"))
  ) |>
  ggplot(aes(x = eval_name, y = eval_accuracy, fill = min_theta_text)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(
    option = "E",
    name = "Alpha",
    #labels = c("Base", "Debiased 3 epochs", "Debiased 10 epochs")
  )+
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  ) +
  scale_x_discrete(
    name = "Evaluation Set",
    #labels = c("dev_matched", "dev_mismatched", "HANS")
  ) +
  facet_wrap(~shallow_model, ncol = 1) +
  theme_bw(16)
ggsave(str_c(dir, "plots/annealing.png"), width = 8, height = 8, units = "in", dpi = 300)
