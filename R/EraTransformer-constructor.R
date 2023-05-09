
pad_sequence <- function(inputs, params) {
  if (is.null(params$max_len)) {
    params$max_len <- max(sapply(inputs, function(input) dim(input)[1]))
  }
  padded_inputs <- list()
  masks <- list()
  for (input in inputs) {
    pad_len <- params$max_len - dim(input)[1]
    padded_input <- torch::nnf_pad(input, c(0, 0, 0, pad_len), value = params$padding_value)
    mask <- torch::torch_ones(c(dim(input)[1], 1), dtype=torch_float())
    masks[[length(masks) + 1]] <- torch_cat(list(mask, torch_zeros(c(pad_len, 1), dtype=torch_float())), dim=1)
    padded_inputs[[length(padded_inputs) + 1]] <- padded_input
  }
  return(list(torch_stack(padded_inputs), torch_stack(masks)))
}

convert_to_torch <- function(era, data, target_names, pred_names, params) {

  inputs <- torch::torch_tensor(as.matrix(data[pred_names]), dtype = torch::torch_int(), device = torch_device(params$device))

  labels <- torch::torch_tensor(as.matrix(data[target_names]), device = torch_device(params$device))

  pad_inputs <- pad_sequence(list(inputs), params)

  padded_inputs <- pad_inputs[[1]]
  masks_inputs <- pad_inputs[[2]]

  pad_labels <- pad_sequence(list(labels), params)

  padded_labels <- pad_labels[[1]]
  masks_labels <- pad_labels[[2]]

  return(list(
    era = list(
      padded_inputs,
      padded_labels,
      masks_inputs
    )
  ))
}

get_era2data <- function(df, target_names, pred_names, params) {

  df <- df %>%
    nest_by.(date)

  future::plan(multisession, workers = params$nthreads)

  era2data <- map2(df$date, df$data, convert_to_torch, target_names, pred_names, params)
                          # .options = furrr_options(globals = TRUE, packages = "torch"))

  future::plan(sequential)



  return(era2data)
}

PositionalEncoding <- nn_module(
  classname = "PositionalEncoding",
  initialize = function(d_model, max_len=5000) {

    self$dropout <- nn_dropout(p=0.1)
    pe <- torch_zeros(max_len, d_model)
    position <- torch_unsqueeze(torch_arange(1, max_len, dtype=torch_float()), 2)
    div_term <- torch_exp(
      torch_arange(1, d_model, 2)
      * (-torch_log(torch_tensor(10000)) / d_model)
    )
    pe[, seq(1, d_model, 2)] <- torch_sin(position * div_term)
    pe[, seq(2, d_model, 2)] <- torch_cos(position * div_term)
    pe <- torch::torch_transpose(torch::torch_unsqueeze(pe,1),1,2)
    self$register_buffer("pe", pe)
  },

  forward = function(x) {
    x <- x + self$pe[seq_len(dim(x)[1]), , drop=FALSE]
    return(self$dropout$forward(x))
  }

)

FeedForwardLayer <- nn_module(
  classname = "FeedForwardLayer",
  initialize = function(d_model, d_ff=128, dropout=0.1) {

    self$linear_1 <- nn_linear(d_model, d_ff)
    self$dropout <- nn_dropout(dropout)
    self$linear_2 <- nn_linear(d_ff, d_model)
  },

  forward = function(x) {
    x <- self$dropout$forward(nnf_relu(self$linear_1$forward(x)))
    x <- self$linear_2$forward(x)
    return(x)
  }

)

SelfAttention <- nn_module(
  classname = "SelfAttention",
  initialize = function(d_model, dropout=0.1) {
    super$initialize()
    self$linear_k <- nn_linear(d_model, d_model)
    self$linear_q <- nn_linear(d_model, d_model)
    self$linear_v <- nn_linear(d_model, d_model)
    self$dim <- d_model
    self$dropout <- nn_dropout(dropout)
  },

  forward = function(inputs, mask=NULL, params) {
    k <- self$linear_k$forward(inputs)
    q <- self$linear_q$forward(inputs)
    v <- self$linear_v$forward(inputs)
    n <- torch_sqrt(torch_tensor(self$dim, dtype=torch_float32(), device = torch_device(params$device)))

    scores <- torch_bmm(q, k$transpose(2, 3)) / n
    if (!is.null(mask)) {
      scores <- scores$masked_fill(mask == 0, -Inf)
    }

    attention_weights <- nnf_softmax(scores, dim=1)
    attention_weights <- self$dropout$forward(attention_weights)

    attention_weights <- torch_bmm(attention_weights, v) / n

    return(attention_weights)
  }

)

MultiHeadLinearAttention <- nn_module(
  classname = "MultiHeadLinearAttention",
  initialize = function(d_model, num_heads) {
    super$initialize()
    self$d_model <- d_model
    self$num_heads <- num_heads
    # SelfAttention$debug("forward")
    self$attention <- SelfAttention(d_model, dropout=0.15)
    self$layers <- nn_module_list(
      lapply(seq_len(num_heads), function(x) {
        nn_linear(d_model, d_model)
      })
    )
    self$fc <- nn_linear(num_heads * d_model, d_model)
  },

  forward = function(inputs, mask=NULL, params) {
    x <- inputs
    head_outputs <- list()
    for (layer in seq_len(self$num_heads)) {
      attention_weights <- self$attention$forward(x, mask, params)
      head_output <- x * attention_weights
      head_output <- self$layers[[layer]]$forward(head_output)
      head_output <- torch_relu(head_output)
      head_outputs[[length(head_outputs) + 1]] <- head_output
    }

    concatenated <- torch_cat(head_outputs, dim=-1)
    output <- self$fc$forward(concatenated)

    return(output)
  }
)

TransformerEncoder <- nn_module(
  classname = "TransformerEncoder",
    initialize = function(
    input_dim,
    d_model,
    output_dim,
    num_heads,
    num_layers,
    dropout_prob=0.15,
    max_len=5000
    ) {
      self$input_dim <- input_dim
      self$output_dim <- output_dim
      self$num_heads <- num_heads
      self$num_layers <- num_layers
      self$dropout_prob <- dropout_prob
      self$d_model <- d_model

      self$positional_encoding <- PositionalEncoding(d_model, max_len)
      self$attention <- MultiHeadLinearAttention(d_model, num_heads)
      self$fc <- nn_sequential(
        nn_linear(d_model, d_model)
      )


      self$layers <- nn_module_list(
        lapply(seq_len(num_layers), function(x) {
          nn_sequential(FeedForwardLayer(d_model=d_model))
        })
      )

      self$mapper <- nn_sequential(
        nn_linear(input_dim, d_model),
        nn_linear(d_model, d_model)
      )
    },

    forward = function(inputs, mask=NULL, params) {
      x <- self$mapper$forward(inputs)
      pe <- self$positional_encoding$forward(x)
      x <- x + pe # works without PE as well
      for (layer in seq_len(self$num_layers)) {
        attention_weights <- self$attention$forward(x, mask, params)
        x <- nnf_layer_norm(x, c(dim(x)[2],dim(x)[3]))
        x <- nnf_dropout(x, p=self$dropout_prob)

        op <- self$layers[[layer]]$forward(x)
        x <- x + op
        x <- nnf_layer_norm(x, c(dim(x)[2],dim(x)[3]))
        x <- nnf_dropout(x, p=self$dropout_prob)
      }

      outputs <- self$fc$forward(x)
      return(outputs)
    }
)

Transformer <- nn_module(
  classname = "Transformer",
  initialize = function(
    input_dim,
    d_model,
    output_dim,
    num_heads,
    num_layers,
    dropout_prob=0.15,
    max_len=5000,
    params
  ) {
    self$input_dim <- input_dim
    self$output_dim <- output_dim
    self$num_heads <- num_heads
    self$num_layers <- num_layers
    self$dropout_prob <- dropout_prob
    self$d_model <- d_model

    self$encoder <- TransformerEncoder(
      input_dim=input_dim,
      d_model=d_model,
      output_dim=output_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      max_len=max_len
    )
    self$fc <- nn_sequential(
      nn_linear(d_model, d_model / 2),
      nn_selu(),
      nn_linear(d_model / 2, self$output_dim),
      nn_sigmoid()
    )
  },

  forward = function(inputs, mask=NULL, params) {
    emb <- self$encoder$forward(inputs, mask, params)
    outputs <- self$fc$forward(emb)
    return(outputs)
  }


)

pearsonr <- function(x, y) {
  mx <- torch::torch_mean(x)
  my <- torch::torch_mean(y)
  xm <- x - mx
  ym <- y - my
  r_num <- torch_sum(xm * ym)
  r_den <- torch_sqrt(torch_sum(xm ** 2) * torch_sum(ym ** 2))
  r <- r_num / r_den
  return(r)
}

calculate_loss <- function(outputs, padded_labels, masks_inputs, padded_inputs=NULL, target_weight_softmax=NULL) {
  # MSE on all targets; additionally, on primary target
  if (!is.null(target_weight_softmax)) {
    mse <- torch::nnf_mse_loss(
      outputs * masks_inputs * target_weight_softmax,
      padded_labels * masks_inputs * target_weight_softmax
    ) * 0.1
  } else {
    mse <- torch::nnf_mse_loss(outputs * masks_inputs, padded_labels * masks_inputs) * 0.1
  }

  mse <- mse + torch::nnf_mse_loss(outputs[, 1] * masks_inputs, padded_labels[, 1] * masks_inputs)

  # Corr with only primary target; adjust as needed

  corr <- pearsonr(
    outputs[1][, 1][torch::torch_nonzero(masks_inputs$view(-1))]$view(c(-1, 1)),
    padded_labels[1][, 1][torch::torch_nonzero(masks_inputs$view(-1))]$view(c(-1, 1))
  )

  loss <- mse - corr #+ some_complex_constraints
  return(list(loss, mse, corr))
}

train_on_batch <- function(transformer, optimizer, batch, params) {
  padded_inputs <- batch[[1]]$to(device=torch_device(params$device))
  padded_labels <- batch[[2]]$to(device=torch_device(params$device))
  masks_inputs <- batch[[3]]$to(device=torch_device(params$device))

  optimizer$zero_grad()

  outputs <- transformer$forward(padded_inputs / 4, masks_inputs, params)

  target_weight_softmax <- NULL
  #random_weights <- torch_rand(dim(padded_labels)[[2]], device=device)
  #target_weight_softmax <- torch_softmax(random_weights)

  loss_mse_corr <- calculate_loss(outputs, padded_labels, masks_inputs, target_weight_softmax=target_weight_softmax)
  loss <- loss_mse_corr[[1]]
  mse <- loss_mse_corr[[2]]
  corr <- loss_mse_corr[[3]]
  loss$backward()
  optimizer$step()
  return(list(loss$item(), mse$item(), corr$item()))
}

evaluate_on_batch <- function(transformer, batch, params) {
  padded_inputs <- batch[[1]]$to(device=torch_device(params$device))
  padded_labels <- batch[[2]]$to(device=torch_device(params$device))
  masks_inputs <- batch[[3]]$to(device=torch_device(params$device))

  outputs <- transformer$forward(padded_inputs / 4, masks_inputs, params)
  loss_mse_corr <- calculate_loss(outputs, padded_labels, masks_inputs)
  loss <- loss_mse_corr[[1]]
  mse <- loss_mse_corr[[2]]
  corr <- loss_mse_corr[[3]]
  return(list(loss$item(), mse$item(), corr$item()))
}


train_model <- function(transformer, optimizer, train_loader, val_loader, params) {
  for (epoch in 1:params$epochs) {
    total_loss <- c()
    total_corr <- c()

    if (params$verbose) {
      cat(paste0("\nEPOCH: ", epoch, "/", params$epochs))
    }

    for (era_num in 1:length(train_loader)) {
      batch <- train_loader[[era_num]]$era
      res <- train_on_batch(transformer, optimizer, batch, params)
      loss <- res[[1]]
      mse <- res[[2]]
      corr <- res[[3]]
      total_loss <- c(total_loss, loss)
      total_corr <- c(total_corr, corr)
    }
    if (params$verbose) {
      cat(
        paste0(
          "Train Loss: ",
          round(mean(total_loss), 4),
          " | Train Corr: ",
          round(mean(total_corr), 4)
        )
      )
    }
    transformer$eval()
    total_loss <- c()
    total_corr <- c()
    for (era_num in 1:length(val_loader)) {
      batch <- val_loader[[era_num]]$era
      res <- evaluate_on_batch(transformer, batch, params)
      loss <- res[[1]]
      mse <- res[[2]]
      corr <- res[[3]]
      total_loss <- c(total_loss, loss)
      total_corr <- c(total_corr, corr)
    }

    if (params$verbose) {
      cat(
        paste0(
          "Val Loss: ",
          round(mean(total_loss), 4),
          " | Val Corr: ",
          round(mean(total_corr), 4)
        )
      )
    }
    torch::cuda_empty_cache()
    invisible(gc())

    patience_counter <- 0
    # Early-stopping checks
    if (params$early_stopping && params$early_stopping_monitor=="valid_loss") {
      current_loss <- mean(total_loss)
    } else {
      current_loss <- mean(total_corr)
    }
    if (params$early_stopping && epoch > 1) {
      # compute relative change, and compare to best_metric
      change <- (current_loss - best_metric) / abs(current_loss)
      if (change > params$early_stopping_tolerance){
        patience_counter <- patience_counter + 1
        if (patience_counter >= params$early_stopping_patience){
          if (params$verbose)
            rlang::inform(sprintf("Early stopping at epoch %03d", epoch))
          break
        }
      } else {
        # reset the patience counter
        best_metric <- current_loss
        patience_counter <- 0L
      }
    }
    if (params$early_stopping && epoch == 1) {
      # initialise best_metric
      best_metric <- current_loss
    }

  }

  torch::torch_save(transformer$state_dict(), "transformer.pth")

  return(transformer)
}

#' @export
transformer_params <- function(
  epochs = 10,
  learning_rate = 1e-4,
  padding_value = -1,
  max_len = 6000,
  feature_dim = NULL,
  hidden_dim = 128,
  output_dim = NULL,
  num_heads = 2,
  num_layers = 2,
  nthreads = 1,
  validation = 0.1,
  verbose = TRUE,
  early_stopping = FALSE,
  early_stopping_monitor = "valid_loss",
  early_stopping_tolerance = 0,
  early_stopping_patience = 0L,
  device = "cuda"){
  list(
    epochs = epochs,
    learning_rate = learning_rate,
    padding_value = padding_value,
    max_len = max_len,
    feature_dim = feature_dim,
    hidden_dim = hidden_dim,
    output_dim = output_dim,
    num_heads = num_heads,
    num_layers = num_layers,
    nthreads = nthreads,
    validation = validation,
    verbose = verbose,
    early_stopping = early_stopping,
    early_stopping_monitor = early_stopping_monitor,
    early_stopping_tolerance = early_stopping_tolerance,
    early_stopping_patience = early_stopping_patience,
    device = device
  )
}

model_to_raw <- function (model) {
  con <- rawConnection(raw(), open = "wr")
  torch::torch_save(model, con)
  on.exit({
    close(con)
  }, add = TRUE)
  r <- rawConnectionValue(con)
  r}


new_EraTransformer <- function(transformer, blueprint, params) {

  transformer <- model_to_raw(transformer)

  hardhat::new_model(transformer = transformer, blueprint = blueprint, params = params, class = "EraTransformer")
}
