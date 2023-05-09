#' Fit a `EraTransformer`
#'
#' `EraTransformer()` fits a model.
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'     created from [recipes::recipe()].
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 numeric column.
#'   * A __matrix__ with 1 numeric column.
#'   * A numeric __vector__.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#' and the predictor terms on the right-hand side.
#'
#' @param ... Not currently used, but required for extensibility.
#'
#' @return
#'
#' A `EraTransformer` object.
#'
#' @examples
#' predictors <- mtcars[, -1]
#' outcome <- mtcars[, 1]
#'
#' # XY interface
#' mod <- EraTransformer(predictors, outcome)
#'
#' # Formula interface
#' mod2 <- EraTransformer(mpg ~ ., mtcars)
#'
#' # Recipes interface
#' library(recipes)
#' rec <- recipe(mpg ~ ., mtcars)
#' rec <- step_log(rec, disp)
#' mod3 <- EraTransformer(rec, mtcars)
#'
#' @export
EraTransformer <- function(x, ...) {
  UseMethod("EraTransformer")
}

#' @export
#' @rdname EraTransformer
EraTransformer.default <- function(x, ...) {
  stop("`EraTransformer()` is not defined for a '", class(x)[1], "'.", call. = FALSE)
}

# XY method - data frame

#' @export
#' @rdname EraTransformer
EraTransformer.data.frame <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  EraTransformer_bridge(processed, ...)
}

# XY method - matrix

#' @export
#' @rdname EraTransformer
EraTransformer.matrix <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  EraTransformer_bridge(processed, ...)
}

# Formula method

#' @export
#' @rdname EraTransformer
EraTransformer.formula <- function(formula, data, ...) {
  processed <- hardhat::mold(formula, data)
  EraTransformer_bridge(processed, ...)
}

# Recipe method

#' @export
#' @rdname EraTransformer
EraTransformer.recipe <- function(x, data, ...) {
  processed <- hardhat::mold(x, data)
  EraTransformer_bridge(processed, ...)
}

# ------------------------------------------------------------------------------
# Bridge

EraTransformer_bridge <- function(processed,...) {

  dots <- rlang::list2(...)

  params <- transformer_params()

  for (param in names(dots)) {

    if (!is.null(dots[[param]])) {

      params[[param]] <- dots[[param]]

    }
  }


  target_names <- colnames(processed$outcomes)
  pred_names <- colnames(processed$predictors)

  params$feature_dim <- length(pred_names)
  params$output_dim <- length(target_names)

  data <- processed$extras$roles$date %>%
    bind_cols(processed$outcomes) %>%
    bind_cols(processed$predictors)

  dates <- data %>%
    distinct(date)

  train_dates <- dates %>%
    slice_sample(prop = 1 - params$validation)

  valid_dates <- dates %>%
    filter(!(date %in% train_dates$date))

  train <- data %>%
    filter(date %in% train_dates$date)

  validation <- data %>%
    filter(date %in% valid_dates$date)

  rm(data)
  invisible(gc())

  era2data_train = get_era2data(train, target_names, pred_names, params)
  era2data_validation = get_era2data(validation, target_names, pred_names, params)

  # Transformer$undebug("forward")
  # TransformerEncoder$undebug("forward")
  # MultiHeadLinearAttention$undebug("forward")
  # SelfAttention$undebug("forward")

  transformer = Transformer(
    input_dim=params$feature_dim,
    d_model=params$hidden_dim,
    output_dim=params$output_dim,
    num_heads=params$num_heads,
    num_layers=params$num_layers,
  )

  # # load model from checkpoint
  # if "transformer.pth" in os.listdir():
  #   transformer.load_state_dict(torch.load("transformer.pth"))

  transformer$to(device=torch_device(params$device))
  optimizer = optim_adam(transformer$parameters, lr=params$learning_rate)

  # Number of training iterations
  # Train for longer with low LR

  transformer = train_model(transformer, optimizer, era2data_train, era2data_validation, params)

  new_EraTransformer(
    transformer = transformer,
    blueprint = processed$blueprint,
    params = params
  )
}


# ------------------------------------------------------------------------------
# Implementation

EraTransformer_impl <- function(predictors, outcome) {
  list(coefs = 1)
}
