[Original work](https://github.com/Zeta36/chess-alpha-zero)

# Ressources

- [AlphaZero Cheat Sheet](https://miro.medium.com/max/2000/1*0pn33bETjYOimWjlqDLLNw.png)
- [Lessons from implementing AlphaZero](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191)
- [Real player games .pgn files](https://www.ficsgames.org/download.html) (you can select whole year datasets and average elo score > 2000)
- [Online .pgn viewer](https://chesstempo.com/pgn-viewer.html)

# How to play against AI

- Put your best model in **data/model/** and name it **model_best_weight.h5**.
- Download and extract [Arena Chess GUI](http://www.playwitharena.de/) (no installation needed).
- Start Arena software. From the GUI :
  - Select **Engines** => **Install New Engine** => *select your AlphaZero.bat (or AlphaZero.sh for linux) file* and validate
  - Select **Engines** => **Load Engine** => Select the *AlphaZero.bat* file => Clic **Load** => Clic **OK**

You can now play on the Arena GUI against your AI. You will probably be the white player so you have to play first.
Then the AI will compute its best move (it can take some time depending of what *simulation_num_per_move* you set
inside the config file and whether you use a gpu or cpu).