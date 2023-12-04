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

#### Data set Analysis ####
eval_names <- c("dev_matched", "dev_mismatched", "hans", "mnli_train")
eval_locs <- c("mnli_on_dev_matched", "mnli_on_dev_mismatched", "mnli_on_hans", "mnli_on_train") %>%
  str_c(dir, "evals/", ., "/eval_predictions.jsonl")

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
  select(-predicted_label)

df_hans |> 
  group_by(heuristic, gold_label) |> 
  summarize(
    accuracy = mean(correct)
  ) |> 
  ggplot(aes(x = heuristic, y = accuracy, fill = gold_label)) +
  geom_col(position = "dodge") +
  scale_fill_colorblind(
    name = "Gold Label",
    labels = c("Entailment", "Non-entailment")
  ) + 
  scale_y_continuous(
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent
  ) +
  scale_x_discrete(
    name = "Heuristic",
    # Capitalize first letter
    labels = function(x) str_to_title(x)
  ) +
  cowplot::theme_minimal_hgrid()

ggsave(str_c(dir, "plots/mnli_on_hans.png"), width = 8, height = 6, units = "in", dpi = 300)


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
  mutate(eval_name = factor(eval_name, levels = c("mnli_train", "dev_matched", "dev_mismatched", "hans")))

p1 <- df |>
  group_by(eval_name) |> 
  summarize(
    accuracy = mean(correct)
  ) |>
  ggplot(aes(x = eval_name, y = accuracy)) +
  geom_col(fill = "skyblue") + 
  geom_text(aes(label = scales::percent(accuracy)), vjust = -0.5) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent
  ) +
  scale_x_discrete(
    name = "Evaluation Set",
    # Capitalize first letter
    labels = function(x) str_to_title(x)
  ) +
  cowplot::theme_minimal_hgrid()


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
  scale_fill_colorblind(
    name = "Gold Label",
    #labels = c("Entailment", "Non-entailment")
  ) +
  scale_y_continuous(
    limits = c(0, 1.02),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent
  ) +
  scale_x_discrete(
    name = "Evaluation Set",
    # Capitalize first letter
    labels = function(x) str_to_title(x)
  ) +
  cowplot::theme_minimal_hgrid()

patchwork1 <- p1 + p2
patchwork1 + plot_annotation()


# Investigate incorrect on Dev matched and mismatched
df |> 
  filter(eval_name %in% c("dev_matched", "dev_mismatched")) |> 
  filter(correct == 0) |> 
  select(premise, hypothesis, gold_label, pred_label, eval_name) #|> 
  #export(str_c(dir, "evals/incorrect_dev.csv"))

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

df_syn_metrics |>
  filter(!is.na(acc)) |> 
  bind_rows(df_start_values) |> 
  ggplot(aes(x = epoch, y = acc, color = eval_set)) +
  #ignore na when drawing line
  geom_line() +
  geom_point() +
  facet_wrap(~`training samples text`, nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  )  +
  scale_color_brewer(
    palette = "Set1",
    name = "Evaluation Set",
    labels = c("MNLI-Antibias", "MNLI-Bias", "MNLI")
  ) +
  scale_x_continuous(
    name = "Epoch",
    breaks = seq(0, 10, 2)
  ) +
  theme_bw(16)+
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16),
    strip.text = element_text(size = 16)
  )
  
df_syn_metrics |>
  filter(!is.na(acc)) |> 
  bind_rows(df_start_values) |> 
  ggplot(aes(x = epoch, y = c_above_90, color = eval_set)) +
  #ignore na when drawing line
  geom_line() +
  geom_point() +
  facet_wrap(~`training samples text`, nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = 0),
    name = "Accuracy",
    labels = scales::percent,
    breaks = seq(0, 1, 0.1)
  )  +
  scale_color_brewer(
    palette = "Set1",
    name = "Evaluation Set",
    labels = c("MNLI-Antibias", "MNLI-Bias", "MNLI")
  ) +
  scale_x_continuous(
    name = "Epoch",
    breaks = seq(0, 10, 2)
  ) +
  theme_bw(16)+
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16),
    strip.text = element_text(size = 16)
  )
  

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


#### Annealing ####
eval_annealed_theta <- c(50, 60, 70, 80, 90)
eval_annealed_dev_matched <- c(0.587, 0.517, 0.443, 0.373, 0.305)
eval_annealed_dev_mismatched <- c(0.605,0.517, 0.444, 0.373, 0.309)
eval_annealed_hans <- c(0.501, 0.505, 0.507, 0.515, 0.501)

tibble(eval_annealed_theta, eval_annealed_dev_matched, eval_annealed_dev_mismatched, eval_annealed_hans) |> 
  pivot_longer(-eval_annealed_theta, names_to = "eval_name", values_to = "accuracy") |> 
  ggplot(aes(x = eval_annealed_theta, y = accuracy, color = eval_name)) +
  geom_line(size = 1.2) +
  geom_point() +
  scale_x_continuous(
    name = "Min Theta",
    breaks = seq(50, 90, 10)
  ) +
  scale_y_continuous(
    name = "Accuracy",
    limits = c(0, 1),
    expand = expansion(mult = 0),
    labels = scales::percent
  ) +
  scale_color_discrete(
    name = "Evaluation Set",
    labels = function(x) str_to_title(x)
  ) +
  cowplot::theme_minimal_hgrid()
