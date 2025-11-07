# ♟️ PI1 - Predição de Resultados de Partidas de Xadrez (PGN Local)

Este projeto tem como objetivo **analisar partidas de xadrez** em formato PGN e **prever o resultado da partida (vitória das brancas, empate ou vitória das pretas)** usando **técnicas de aprendizado de máquina**.  

O script lê suas partidas exportadas em PGN, extrai variáveis relevantes (como ratings, número de lances e abertura) e treina modelos de classificação.

---

## 🧑‍🎓 Autor

**Aluno:** Matheus Franklin Brasileiro  

---

## 📂 Estrutura do Projeto

📦 AtividadeIndividual
┣ 📜 generate_pi1_chess_pgn.py
┣ 📜 partidas_matheus.pgn # Arquivo PGN local com partidas
┣ 📜 requirements.txt # Dependências do projeto
┣ 📜 README.md # Este arquivo
┗ 📁 pi1_pgn_output/ # Pasta gerada com resultados (relatórios, figuras, modelos)


---

## ⚙️ Como Executar

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/MATHEUSBRr/PI1-Chess-Prediction.git
cd PI1-Chess-Prediction
2️⃣ Criar o ambiente virtual
No Windows:

python3 -m venv venv
source venv/bin/activate
3️⃣ Instalar as dependências

pip install -r requirements.txt
4️⃣ Executar o script
Coloque seu arquivo PGN (ex: partidas_matheus.pgn) na mesma pasta e rode:

python generate_pi1_chess_pgn.py
O script vai:

Ler suas partidas do PGN

Gerar gráficos exploratórios

Treinar Random Forest e XGBoost

Calcular métricas de acurácia e matrizes de confusão

📊 Saídas Geradas
Após a execução, a pasta pi1_pgn_output/ conterá:

📁 figs/

dist_results_pgn.png — Distribuição dos resultados

boxplot_ratingdiff_pgn.png — Diferença de rating por resultado

hist_moves_pgn.png — Distribuição de número de lances

feat_imp_rf_pgn.png — Importância das features

cm_rf_pgn.png — Matriz de confusão do Random Forest

cm_xgb_pgn.png — Matriz de confusão do XGBoost

📁 models/

rf_pipeline_pgn.pkl — Modelo Random Forest salvo

xgb_pipeline_pgn.pkl — Modelo XGBoost salvo

📄 PI1_Predicao_Xadrez_PGN_Matheus.docx
→ Relatório completo com texto, tabelas e figuras

🧠 Principais Tecnologias Utilizadas
Python 3.10+

pandas / numpy — Manipulação e análise de dados

matplotlib — Geração de gráficos

scikit-learn — Pré-processamento e modelos clássicos (RandomForest, GradientBoosting)

xgboost — Modelo de boosting eficiente

python-chess — Leitura e análise de arquivos PGN

joblib — Salvamento dos modelos

⚠️ Observações
O projeto não inclui o ambiente virtual (venv/), pois ele é específico de cada sistema.

O script foi testado com 10 partidas PGN locais e funciona também com bases maiores.

🧩 Próximos Passos (Melhorias Futuras)
Adicionar métricas de tempo por lance e número de blunders.

Integrar diretamente com a API do Chess.com ou Lichess (coleta automática).

Criar versão web para visualização interativa dos resultados.

🏁 Licença
Este projeto é de uso acadêmico e pode ser livremente utilizado com os devidos créditos.
© 2025 - Matheus Franklin Brasileiro