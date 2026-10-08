# 03_var_coint.R
# Analyse VAR, Cointégration et VECM pour BTC et ETH

# 1. Data Loading
cat("=== 1. Chargement des données ===\n")
suppressWarnings({
  suppressMessages(library(tidyverse))
  suppressMessages(library(vars))
  suppressMessages(library(urca))
  suppressMessages(library(tseries))
  suppressMessages(library(tsDyn))
  suppressMessages(library(jsonlite))
  suppressMessages(library(lmtest))
  suppressMessages(library(zoo))
  suppressMessages(library(ggplot2))
  suppressMessages(library(gridExtra))
})

dir.create('results/figures', recursive=TRUE, showWarnings=FALSE)
dir.create('results/tables', recursive=TRUE, showWarnings=FALSE)

df <- read_csv('data/btc_eth_processed.csv', show_col_types = FALSE) %>% mutate(date = as.Date(date))
sp <- fromJSON('data/split.json')
train <- df %>% filter(date <= as.Date(sp$train_end))
test <- df %>% filter(date >= as.Date(sp$test_start) & date <= as.Date(sp$test_end))

# 2. BTC-ETH Relationship & Rolling Correlation
cat("=== 2. Corrélation BTC-ETH ===\n")
# Scatter plot
p_scatter <- ggplot(train, aes(x=btc_ret, y=eth_ret)) + 
  geom_point(alpha=0.5) + 
  geom_smooth(method="lm", color="red") +
  theme_minimal() + 
  ggtitle("Nuage de points des rendements (BTC vs ETH)")
ggsave('results/figures/03_scatter_ret_R.png', p_scatter, width=8, height=6)

pearson_cor <- cor(train$btc_ret, train$eth_ret, use="complete.obs")
cat("Corrélation de Pearson (rendements):", pearson_cor, "\n")
cor_df <- data.frame(Method="Pearson", Correlation=pearson_cor)
write_csv(cor_df, 'results/tables/03_correlation_R.csv')

# Rolling correlation (k=90)
train_comp <- train %>% filter(!is.na(btc_ret) & !is.na(eth_ret))
roll_cor <- rollapply(zoo(cbind(train_comp$btc_ret, train_comp$eth_ret), order.by=train_comp$date), 
                      width=90, 
                      FUN=function(x) cor(x[,1], x[,2], use="complete.obs"), 
                      by.column=FALSE, fill=NA, align="right")
roll_df <- data.frame(date=train_comp$date, roll_cor=coredata(roll_cor))

p_roll <- ggplot(roll_df, aes(x=date, y=roll_cor)) +
  geom_line(color="blue") +
  theme_minimal() +
  ggtitle("Corrélation glissante sur 90 jours (Rendements BTC & ETH)")
ggsave('results/figures/03_correlation_glissante_R.png', p_roll, width=10, height=5)


# 3. Stationarity Tests
cat("=== 3. Tests de stationnarité ===\n")
run_tests <- function(x, name) {
  x <- na.omit(x)
  if(length(x) == 0) return(NULL)
  
  # ADF
  adf_test <- ur.df(x, type="trend", selectlags="AIC")
  adf_stat <- adf_test@teststat[1]
  adf_crit <- adf_test@cval[1, "5pct"]
  
  # KPSS
  kpss_test <- ur.kpss(x, type="mu", use.lag=trunc(3*sqrt(length(x))/13))
  kpss_stat <- kpss_test@teststat
  kpss_crit <- kpss_test@cval[1, "5pct"]
  
  data.frame(
    Variable = name,
    ADF_Stat = adf_stat,
    ADF_Crit_5pct = adf_crit,
    ADF_Stationary = adf_stat < adf_crit,
    KPSS_Stat = kpss_stat,
    KPSS_Crit_5pct = kpss_crit,
    KPSS_Stationary = kpss_stat < kpss_crit
  )
}

train_na <- na.omit(train %>% select(btc_logp, eth_logp, btc_ret, eth_ret))
stat_res <- bind_rows(
  run_tests(train_na$btc_logp, "btc_logp"),
  run_tests(train_na$eth_logp, "eth_logp"),
  run_tests(train_na$btc_ret, "btc_ret"),
  run_tests(train_na$eth_ret, "eth_ret"),
  run_tests(diff(train_na$btc_logp), "diff(btc_logp)"),
  run_tests(diff(train_na$eth_logp), "diff(eth_logp)")
)
write_csv(stat_res, 'results/tables/03_stationnarite_multi_R.csv')

# 4. Cointegration Tests
cat("=== 4. Tests de cointégration ===\n")
# Engle-Granger
eg_lm <- lm(btc_logp ~ eth_logp, data=train_na)
eg_resid <- residuals(eg_lm)
eg_adf <- ur.df(eg_resid, type="none", selectlags="AIC")

# Johansen
jo_data <- train_na %>% select(btc_logp, eth_logp)
jo_trace <- ca.jo(jo_data, type="trace", ecdet="const", K=2)
jo_eigen <- ca.jo(jo_data, type="eigen", ecdet="const", K=2)

coint_res <- data.frame(
  Test = c("Engle-Granger ADF Stat", "Johansen Trace (r=0)", "Johansen Eigen (r=0)"),
  Statistic = c(eg_adf@teststat[1], jo_trace@teststat[2], jo_eigen@teststat[2]),
  Crit_5pct = c(eg_adf@cval[1, "5pct"], jo_trace@cval[2, "5pct"], jo_eigen@cval[2, "5pct"])
)
write_csv(coint_res, 'results/tables/03_cointegration_R.csv')


# 5. VAR Model
cat("=== 5. Modèle VAR ===\n")
var_data <- train_na %>% select(btc_ret, eth_ret)
var_select <- VARselect(var_data, lag.max=20, type="const")
optimal_lag <- var_select$selection["AIC(n)"]

var_sel_df <- data.frame(Criteria = names(var_select$selection), Lag = as.numeric(var_select$selection))
write_csv(var_sel_df, 'results/tables/03_var_selection_retards_R.csv')

var_model <- VAR(var_data, p=optimal_lag, type="const")
cat("Retard optimal choisi (AIC) :", optimal_lag, "\n")


# 6. Granger Causality
cat("=== 6. Causalité au sens de Granger ===\n")
g1 <- causality(var_model, cause="btc_ret")
g2 <- causality(var_model, cause="eth_ret")

granger_res <- data.frame(
  Direction = c("BTC -> ETH", "ETH -> BTC"),
  P_Value = c(g1$Granger$p.value, g2$Granger$p.value)
)
write_csv(granger_res, 'results/tables/03_granger_R.csv')
print(granger_res)


# 7. VECM
cat("=== 7. Modèle VECM ===\n")
jo_vecm <- ca.jo(jo_data, type="trace", K=optimal_lag, ecdet="const")
vecm_model <- cajorls(jo_vecm, r=1)

# Le vecteur de cointégration (beta)
cat("Vecteur de cointégration (beta) :\n")
print(vecm_model$beta)

# Les termes de correction d'erreur (alpha) = coefficient de ect1
ect_coefs <- vecm_model$rlm$coefficients["ect1",]
cat("Termes de correction d'erreur (alpha) :\n")
print(ect_coefs)

vecm_df <- data.frame(
  Variable = names(ect_coefs),
  ErrorCorrectionTerm = as.numeric(ect_coefs)
)
write_csv(vecm_df, 'results/tables/03_vecm_summary_R.csv')


# 8. Forecasts & Evaluation
cat("=== 8. Prévisions ===\n")
h <- nrow(test)

# VAR Forecast
var_pred <- predict(var_model, n.ahead=h)
pred_btc_var <- var_pred$fcst$btc_ret[, "fcst"]
pred_eth_var <- var_pred$fcst$eth_ret[, "fcst"]

# Convert VECM to VAR for forecasting
vec2var_model <- vec2var(jo_vecm, r=1)
vecm_pred <- predict(vec2var_model, n.ahead=h)
# Predictions are in levels (logp), need to difference to get returns or compare levels
pred_btc_vecm_log <- vecm_pred$fcst$btc_logp[, "fcst"]
pred_eth_vecm_log <- vecm_pred$fcst$eth_logp[, "fcst"]

# Calculate errors for VAR (on returns)
calc_metrics <- function(actual, predicted) {
  actual <- na.omit(actual)
  predicted <- predicted[1:length(actual)]
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted)/actual)) * 100
  c(RMSE=rmse, MAE=mae, MAPE=mape)
}

test_na <- test %>% filter(!is.na(btc_ret) & !is.na(eth_ret))
actual_btc_ret <- test_na$btc_ret
actual_eth_ret <- test_na$eth_ret

metrics_var_btc <- calc_metrics(actual_btc_ret, pred_btc_var)
metrics_var_eth <- calc_metrics(actual_eth_ret, pred_eth_var)

metrics_df <- data.frame(
  Model = c("VAR_BTC_ret", "VAR_ETH_ret"),
  RMSE = c(metrics_var_btc["RMSE"], metrics_var_eth["RMSE"]),
  MAE = c(metrics_var_btc["MAE"], metrics_var_eth["MAE"]),
  MAPE = c(metrics_var_btc["MAPE"], metrics_var_eth["MAPE"])
)
write_csv(metrics_df, 'results/tables/03_metriques_var_R.csv')

# Save forecasts
write_csv(data.frame(date=test_na$date, actual=actual_btc_ret, pred=pred_btc_var[1:length(actual_btc_ret)]), 'results/tables/forecasts_var_btc_R.csv')
write_csv(data.frame(date=test_na$date, actual=actual_eth_ret, pred=pred_eth_var[1:length(actual_eth_ret)]), 'results/tables/forecasts_var_eth_R.csv')

# Plot VAR Forecasts
plot_var_btc <- ggplot() +
  geom_line(aes(x=test_na$date, y=actual_btc_ret, color="Actual")) +
  geom_line(aes(x=test_na$date, y=pred_btc_var[1:length(actual_btc_ret)], color="VAR Pred")) +
  theme_minimal() + ggtitle("Prévisions VAR - BTC Returns")

plot_var_eth <- ggplot() +
  geom_line(aes(x=test_na$date, y=actual_eth_ret, color="Actual")) +
  geom_line(aes(x=test_na$date, y=pred_eth_var[1:length(actual_eth_ret)], color="VAR Pred")) +
  theme_minimal() + ggtitle("Prévisions VAR - ETH Returns")

p_var_grid <- grid.arrange(plot_var_btc, plot_var_eth, ncol=1)
ggsave('results/figures/03_previsions_var_R.png', p_var_grid, width=10, height=8)

# Plot VECM Forecasts (Levels)
test_log_na <- test %>% filter(!is.na(btc_logp) & !is.na(eth_logp))
plot_vecm_btc <- ggplot() +
  geom_line(aes(x=test_log_na$date, y=test_log_na$btc_logp, color="Actual")) +
  geom_line(aes(x=test_log_na$date, y=pred_btc_vecm_log[1:nrow(test_log_na)], color="VECM Pred")) +
  theme_minimal() + ggtitle("Prévisions VECM - BTC Log Price")

plot_vecm_eth <- ggplot() +
  geom_line(aes(x=test_log_na$date, y=test_log_na$eth_logp, color="Actual")) +
  geom_line(aes(x=test_log_na$date, y=pred_eth_vecm_log[1:nrow(test_log_na)], color="VECM Pred")) +
  theme_minimal() + ggtitle("Prévisions VECM - ETH Log Price")

p_vecm_grid <- grid.arrange(plot_vecm_btc, plot_vecm_eth, ncol=1)
ggsave('results/figures/03_previsions_vecm_R.png', p_vecm_grid, width=10, height=8)


# 9. IRF
cat("=== 9. Fonctions de réponse impulsionnelle (IRF) ===\n")
irf_model <- irf(var_model, n.ahead=10, boot=TRUE, runs=100)
png('results/figures/03_irf_R.png', width=800, height=600)
plot(irf_model)
dev.off()


# 10. FEVD
cat("=== 10. Décomposition de la variance (FEVD) ===\n")
fevd_model <- fevd(var_model, n.ahead=10)
png('results/figures/03_fevd_R.png', width=800, height=600)
plot(fevd_model)
dev.off()

cat("Terminé avec succès.\n")
