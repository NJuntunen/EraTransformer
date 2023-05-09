#' Predict from a `EraTransformer`
#'
#' @param object A `EraTransformer` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#'
#' @param type A single character. The type of predictions to generate.
#' Valid options are:
#'
#' - `"numeric"` for numeric predictions.
#'
#' @param ... Not used, but required for extensibility.
#'
#' @return
#'
#' A tibble of predictions. The number of rows in the tibble is guaranteed
#' to be the same as the number of rows in `new_data`.
#'
#' @examples
#' train <- mtcars[1:20,]
#' test <- mtcars[21:32, -1]
#'
#' # Fit
#' mod <- EraTransformer(mpg ~ cyl + log(drat), train)
#'
#' # Predict, with preprocessing
#' predict(mod, test)
#'
#' @export
predict.EraTransformer <- function(object, new_data, type = "numeric", ...) {
  forged <- hardhat::forge(new_data, object$blueprint)
  rlang::arg_match(type, valid_EraTransformer_predict_types())
  predict_EraTransformer_bridge(type, object, forged)
}

valid_EraTransformer_predict_types <- function() {
  c("numeric")
}

# ------------------------------------------------------------------------------
# Bridge

predict_EraTransformer_bridge <- function(type, model, processed) {

  target_names <- colnames(processed$outcomes)
  pred_names <- colnames(processed$predictors)

  data <- processed$extras$roles$date %>%
    bind_cols(processed$outcomes) %>%
    bind_cols(processed$predictors)

  predict_function <- get_EraTransformer_predict_function(type)
  predictions <- predict_function(model, data, target_names, pred_names)

  hardhat::validate_prediction_size(predictions, data)

  predictions
}

get_EraTransformer_predict_function <- function(type) {
  switch(
    type,
    numeric = predict_EraTransformer_numeric
  )
}

# ------------------------------------------------------------------------------
# Implementation

predict_EraTransformer_numeric <- function(model, data, target_names, pred_names) {

  model$transformer <- torch_load(model$transformer)

  model$transformer$to(device=torch_device(model$params$device))
  model$transformer$eval()

  model$params$max_len <- NULL

  data <- data %>%
    group_by(date) %>%
    group_split()

  preds <- list()

  for (era in 1:length(data)) {

    inputs <- torch::torch_tensor(as.matrix(data[[era]][pred_names]), dtype = torch::torch_int())

    pad_inputs <- pad_sequence(list(inputs), model$params)

    padded_inputs <- pad_inputs[[1]]$to(device=torch_device(model$params$device))

    padded_masks <- pad_inputs[[2]]$to(device=torch_device(model$params$device))

    outputs <- as_array(model$transformer(padded_inputs/4.0, padded_masks, model$params)[1][torch_nonzero(padded_masks$view(-1))]$squeeze(2)$cpu())

    preds[[era]] <- outputs

  }

  suppressMessages(
    predictions <- map_df(preds, as_tibble, .name_repair = "unique")
  )

  num_cols <- ncol(predictions)

  if (num_cols == 1){

    new_colnames <- ".pred"

  }else{

    new_colnames <- paste0("pred_", seq_len(num_cols))

  }

  colnames(predictions) <- new_colnames


  return(predictions)
}















