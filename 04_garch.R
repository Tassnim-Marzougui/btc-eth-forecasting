# 04_garch.R

# Création des dossiers si nécessaires
dir.create('results/figures', recursive=TRUE, showWarnings=FALSE)
dir.create('results/tables', recursive=TRUE, showWarnings=FALSE)

cat("## 1. Data Loading\n")
suppressWarnings(suppressMessages({
  library(tidyverse)
  library(rugarch)
  library(FinTS)
  library(tseries)
  library(jsonlite)
  library(lmtest)
  library(zoo)
  library(gridExtra)
}))

df <- read_csv('data/btc_eth_processed.csv', show_col_types = FALSE) %>% 
  mutate(date = as.Date(date)) %>%
  drop_na(btc_ret, eth_ret) # Assurer qu'il n'y a pas de NA

sp <- fromJSON('data/split.json')
train <- df %>% filter(date <= as.Date(sp$train_end))
test <- df %>% filter(date >= as.Date(sp$test_start) & date <= as.Date(sp$test_end))

# Extraire les séries temporelles de rendements
btc_ret_train <- train$btc_ret
eth_ret_train <- train$eth_ret
date_train <- train$date

cat("## 2. ARCH-LM Test\n")
arch_btc <- ArchTest(btc_ret_train, lags=12, demean=TRUE)
arch_eth <- ArchTest(eth_ret_train, lags=12, demean=TRUE)

arch_results <- data.frame(
  Asset = c("BTC", "ETH"),
  Statistic = c(arch_btc$statistic, arch_eth$statistic),
  P_value = c(arch_btc$p.value, arch_eth$p.value)
)
print(arch_results)
write_csv(arch_results, 'results/tables/04_arch_lm_R.csv')

cat("## 3. GARCH(1,1) for BTC\n")
# Fonction pour ajuster GARCH(1,1) avec différentes distributions
fit_garch11 <- function(ret, dist) {
  spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                     mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
                     distribution.model = dist)
  fit <- ugarchfit(spec = spec, data = ret, solver = "hybrid")
  return(fit)
}

btc_norm <- fit_garch11(btc_ret_train, "norm")
btc_std <- fit_garch11(btc_ret_train, "std")
btc_sstd <- fit_garch11(btc_ret_train, "sstd")

btc_ic <- data.frame(
  Distribution = c("norm", "std", "sstd"),
  AIC = c(infocriteria(btc_norm)[1], infocriteria(btc_std)[1], infocriteria(btc_sstd)[1]),
  BIC = c(infocriteria(btc_norm)[2], infocriteria(btc_std)[2], infocriteria(btc_sstd)[2])
)
write_csv(btc_ic, 'results/tables/04_garch_selection_btc_R.csv')

cat("## 4. GARCH(1,1) for ETH\n")
eth_norm <- fit_garch11(eth_ret_train, "norm")
eth_std <- fit_garch11(eth_ret_train, "std")
eth_sstd <- fit_garch11(eth_ret_train, "sstd")

eth_ic <- data.frame(
  Distribution = c("norm", "std", "sstd"),
  AIC = c(infocriteria(eth_norm)[1], infocriteria(eth_std)[1], infocriteria(eth_sstd)[1]),
  BIC = c(infocriteria(eth_norm)[2], infocriteria(eth_std)[2], infocriteria(eth_sstd)[2])
)
write_csv(eth_ic, 'results/tables/04_garch_selection_eth_R.csv')

cat("## 5. Higher Order GARCH\n")
orders <- list(c(1,1), c(1,2), c(2,1), c(2,2))
compare_orders <- function(ret, asset) {
  results <- data.frame()
  for (o in orders) {
    spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = o),
                       mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")
    fit <- tryCatch(ugarchfit(spec = spec, data = ret, solver = "hybrid"), error=function(e) NULL)
    if (!is.null(fit)) {
      ic <- infocriteria(fit)
      results <- rbind(results, data.frame(Asset=asset, Order=paste0(o[1],",",o[2]), AIC=ic[1], BIC=ic[2]))
    }
  }
  return(results)
}
order_btc <- compare_orders(btc_ret_train, "BTC")
order_eth <- compare_orders(eth_ret_train, "ETH")
order_results <- rbind(order_btc, order_eth)
write_csv(order_results, 'results/tables/04_garch_ordres_R.csv')

cat("## 6. Asymmetric Models\n")
fit_asym <- function(ret, model_name) {
  spec <- ugarchspec(variance.model = list(model = model_name, garchOrder = c(1, 1)),
                     mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")
  fit <- ugarchfit(spec = spec, data = ret, solver = "hybrid")
  return(fit)
}

btc_egarch <- fit_asym(btc_ret_train, "eGARCH")
btc_gjr <- fit_asym(btc_ret_train, "gjrGARCH")
eth_egarch <- fit_asym(eth_ret_train, "eGARCH")
eth_gjr <- fit_asym(eth_ret_train, "gjrGARCH")

asym_results <- data.frame(
  Asset = c("BTC", "BTC", "BTC", "ETH", "ETH", "ETH"),
  Model = c("sGARCH", "eGARCH", "gjrGARCH", "sGARCH", "eGARCH", "gjrGARCH"),
  AIC = c(infocriteria(btc_std)[1], infocriteria(btc_egarch)[1], infocriteria(btc_gjr)[1],
          infocriteria(eth_std)[1], infocriteria(eth_egarch)[1], infocriteria(eth_gjr)[1]),
  BIC = c(infocriteria(btc_std)[2], infocriteria(btc_egarch)[2], infocriteria(btc_gjr)[2],
          infocriteria(eth_std)[2], infocriteria(eth_egarch)[2], infocriteria(eth_gjr)[2])
)
write_csv(asym_results, 'results/tables/04_garch_asymetrique_R.csv')

cat("## 7. Residual Diagnostics\n")
# Choix du meilleur modèle pour la suite: eGARCH(1,1) avec std pour les deux
best_btc <- btc_egarch
best_eth <- eth_egarch

std_res_btc <- residuals(best_btc, standardize=TRUE)
std_res_eth <- residuals(best_eth, standardize=TRUE)

# QQ-plots
png('results/figures/04_qq_residus_btc_R.png', width=800, height=600)
qqnorm(as.numeric(std_res_btc), main="QQ-Plot Résidus Standardisés - BTC")
qqline(as.numeric(std_res_btc), col="red")
invisible(dev.off())

png('results/figures/04_qq_residus_eth_R.png', width=800, height=600)
qqnorm(as.numeric(std_res_eth), main="QQ-Plot Résidus Standardisés - ETH")
qqline(as.numeric(std_res_eth), col="red")
invisible(dev.off())

# ACF
png('results/figures/04_acf_residus_garch_R.png', width=800, height=800)
par(mfrow=c(2,2))
acf(as.numeric(std_res_btc), main="ACF Res Std BTC")
acf(as.numeric(std_res_btc)^2, main="ACF Res Std Carrés BTC")
acf(as.numeric(std_res_eth), main="ACF Res Std ETH")
acf(as.numeric(std_res_eth)^2, main="ACF Res Std Carrés ETH")
invisible(dev.off())

# Tests de diagnostics
lb_btc <- Box.test(std_res_btc, type='Ljung-Box', lag=10)
lb_eth <- Box.test(std_res_eth, type='Ljung-Box', lag=10)
arch_res_btc <- ArchTest(as.numeric(std_res_btc), lags=12, demean=TRUE)
arch_res_eth <- ArchTest(as.numeric(std_res_eth), lags=12, demean=TRUE)

diag_results <- data.frame(
  Asset = c("BTC", "ETH"),
  LjungBox_p = c(lb_btc$p.value, lb_eth$p.value),
  ARCHLM_p = c(arch_res_btc$p.value, arch_res_eth$p.value)
)
write_csv(diag_results, 'results/tables/04_diagnostic_garch_R.csv')

cat("## 8. Conditional Volatility Visualization\n")
vol_btc <- sigma(best_btc)
vol_eth <- sigma(best_eth)
vol_df <- data.frame(
  date = date_train,
  vol_btc = as.numeric(vol_btc),
  vol_eth = as.numeric(vol_eth),
  vol_btc_ann = as.numeric(vol_btc) * sqrt(365),
  vol_eth_ann = as.numeric(vol_eth) * sqrt(365)
)

p1 <- ggplot(vol_df, aes(x=date)) + 
  geom_line(aes(y=vol_btc, color="BTC")) + 
  geom_line(aes(y=vol_eth, color="ETH")) +
  labs(title="Volatilité conditionnelle quotidienne (eGARCH)", y="Sigma") +
  theme_minimal()
ggsave('results/figures/04_volatilite_conditionnelle_R.png', p1, width=10, height=6)

p2 <- ggplot(vol_df, aes(x=date)) + 
  geom_line(aes(y=vol_btc_ann, color="BTC")) + 
  geom_line(aes(y=vol_eth_ann, color="ETH")) +
  labs(title="Volatilité conditionnelle annualisée (eGARCH)", y="Sigma annualisé") +
  theme_minimal()
ggsave('results/figures/04_volatilite_annualisee_R.png', p2, width=10, height=6)


cat("## 9. Forecasting\n")
# Prévisions sur l'ensemble de test en fixant les paramètres
spec_btc <- getspec(best_btc)
setfixed(spec_btc) <- as.list(coef(best_btc))
filter_btc <- ugarchfilter(spec_btc, c(btc_ret_train, test$btc_ret))

spec_eth <- getspec(best_eth)
setfixed(spec_eth) <- as.list(coef(best_eth))
filter_eth <- ugarchfilter(spec_eth, c(eth_ret_train, test$eth_ret))

sigma_all_btc <- sigma(filter_btc)
sigma_all_eth <- sigma(filter_eth)
n_train <- length(btc_ret_train)
sigma_test_btc <- sigma_all_btc[(n_train+1):length(sigma_all_btc)]
sigma_test_eth <- sigma_all_eth[(n_train+1):length(sigma_all_eth)]

fc_df <- data.frame(
  date = test$date,
  pred_vol_btc = as.numeric(sigma_test_btc),
  pred_vol_eth = as.numeric(sigma_test_eth),
  actual_ret_btc = test$btc_ret,
  actual_ret_eth = test$eth_ret
)
# Proxy de la volatilité réalisée par la valeur absolue du rendement
fc_df$actual_vol_btc <- abs(fc_df$actual_ret_btc) 
fc_df$actual_vol_eth <- abs(fc_df$actual_ret_eth)

write_csv(fc_df %>% select(date, pred_vol_btc), 'results/tables/forecasts_garch_btc_R.csv')
write_csv(fc_df %>% select(date, pred_vol_eth), 'results/tables/forecasts_garch_eth_R.csv')

rmse_btc <- sqrt(mean((fc_df$pred_vol_btc - fc_df$actual_vol_btc)^2))
rmse_eth <- sqrt(mean((fc_df$pred_vol_eth - fc_df$actual_vol_eth)^2))

metriques <- data.frame(
  Asset = c("BTC", "ETH"),
  RMSE_Vol = c(rmse_btc, rmse_eth)
)
write_csv(metriques, 'results/tables/04_metriques_garch_R.csv')

p_fc <- ggplot(fc_df, aes(x=date)) +
  geom_line(aes(y=pred_vol_btc, color="Pred BTC Vol")) +
  geom_line(aes(y=pred_vol_eth, color="Pred ETH Vol")) +
  labs(title="Prévisions de volatilité sur ensemble de test", y="Sigma") +
  theme_minimal()
ggsave('results/figures/04_previsions_garch_R.png', p_fc, width=10, height=6)


cat("## 10. VaR\n")
# Calcul de VaR
dist_q1_btc <- qdist("std", p=0.01, shape=coef(best_btc)["shape"])
dist_q5_btc <- qdist("std", p=0.05, shape=coef(best_btc)["shape"])

mu_btc <- fitted(filter_btc)[(n_train+1):length(sigma_all_btc)]
mu_eth <- fitted(filter_eth)[(n_train+1):length(sigma_all_eth)]

fc_df$var1_btc <- mu_btc + sigma_test_btc * dist_q1_btc
fc_df$var5_btc <- mu_btc + sigma_test_btc * dist_q5_btc

dist_q1_eth <- qdist("std", p=0.01, shape=coef(best_eth)["shape"])
dist_q5_eth <- qdist("std", p=0.05, shape=coef(best_eth)["shape"])

fc_df$var1_eth <- mu_eth + sigma_test_eth * dist_q1_eth
fc_df$var5_eth <- mu_eth + sigma_test_eth * dist_q5_eth

viol1_btc <- mean(fc_df$actual_ret_btc < fc_df$var1_btc)
viol5_btc <- mean(fc_df$actual_ret_btc < fc_df$var5_btc)
viol1_eth <- mean(fc_df$actual_ret_eth < fc_df$var1_eth)
viol5_eth <- mean(fc_df$actual_ret_eth < fc_df$var5_eth)

var_violations <- data.frame(
  Asset = c("BTC", "ETH"),
  Violation_1pct = c(viol1_btc, viol1_eth),
  Violation_5pct = c(viol5_btc, viol5_eth)
)
write_csv(var_violations, 'results/tables/04_var_violations_R.csv')

p_var <- ggplot(fc_df, aes(x=date)) +
  geom_line(aes(y=actual_ret_btc, color="Actual Ret BTC"), alpha=0.5) +
  geom_line(aes(y=var1_btc, color="VaR 1%"), linetype="dashed") +
  geom_line(aes(y=var5_btc, color="VaR 5%"), linetype="dashed") +
  labs(title="Rendements BTC et Valeur à Risque (VaR)", y="Rendement") +
  theme_minimal()
ggsave('results/figures/04_var_risk_R.png', p_var, width=10, height=6)

cat("Fin de l'analyse GARCH.\n")
