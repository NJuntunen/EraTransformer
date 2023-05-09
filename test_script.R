pacman::p_load(qs, apila, tidymodels, tidyverse, foreach, iterators, tidytable, torch, future, furrr, EraTransformer)

data <- qread(file.path(get_apila_repo_path(),"Slices", "Set65", "model-Slice01.rds"))

training <- training(data) %>% ungroup

train <- training %>%
  filter_tails(target = "target_1", tail = 0.2) %>%
  ungroup %>%
  select(Id, date, target_1, target_2, target_3, 5:30, -c(cntry, sector))

train_rec <- recipe(train %>% dplyr::slice(0)) %>%
  update_role(everything()) %>%
  update_role(c("target_1"), new_role = "outcome") %>%
  update_role('Id', new_role = "ID") %>%
  update_role('date', new_role = "date") %>%
  step_nzv(all_numeric_predictors()) %>%
  step_mutate(across(where(is.numeric), ~ifelse(is.na(.),-2,.)))

model <- EraTransformer(train_rec, train, epochs = 2, device = "cuda")

qsave(model, "D:/testi_malli.rds")

model <- qread("D:/testi_malli.rds")

test <- testing(data) %>% ungroup

test <- test %>%
    select(names(train))

predict(model, test)


unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

unregister_dopar()









