# Read JSONs and evaluate


# load libraries
library(jsonlite)
library(tidyverse)
library(ggthemes)
library(patchwork)
library(rio)

# Type dir in console
dir <- "D:/OneDrive/MSDS/Natural_Language_Processing/fp-dataset-artifacts/"


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
  scale_color_colorblind(
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
