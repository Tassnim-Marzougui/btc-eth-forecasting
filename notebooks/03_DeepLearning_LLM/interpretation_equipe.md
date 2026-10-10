1. Stationnarité. Les log-prix de BTC et ETH ne sont pas stationnaires (ADF p = 0,35 et 0,41, KPSS rejette la stationnarité), contrairement aux rendements (ADF p < 0,001). On différencie donc une fois. Le BIC retient ARIMA(0,1,0), c’est-à-dire une marche aléatoire.

2. Prévision des prix. Aucun modèle ne bat significativement la marche aléatoire à 10 jours. ARIMA (AIC) a le RMSE le plus bas (3 599 contre 3 607 pour BTC ; 143,0 contre 145,0 pour ETH), mais l’écart est trop faible pour être significatif au test de Diebold-Mariano.

3. Relation BTC–ETH. Les deux actifs sont très corrélés (0,83 sur les rendements) mais ne sont pas cointégrés (Johansen : trace 9,12 < 15,49 ; Engle-Granger p = 0,65). Le VECM n’est donc pas justifié, et il est d’ailleurs significativement moins bon que Naive (p ≈ 0,01). Granger indique que ETH aide à prévoir BTC (p = 0,0006), alors que l’inverse est marginal (p = 0,059). C’est un résultat de prévision, pas de causalité économique.

4. Volatilité. Les effets ARCH sont très significatifs et un GARCH(1,1) suffit à capturer la dynamique (résidus au carré sans autocorrélation). Les termes d’asymétrie ne sont pas significatifs. La volatilité est la partie réellement modélisable, contrairement au niveau des prix.

5. Deep learning. Les courbes de perte sont plates autour de la variance des rendements : LSTM et GRU ne trouvent pas de signal. Leurs prévisions ressemblent à Naive avec une légère dérive (environ +0,09 % par jour sur BTC). Ils ne sont pas significativement différents de Naive, ni entre eux.

6. Limites. Seulement 32 origines (tests peu puissants), une seule période de test, des réseaux univariés, une seule graine