# %%
# Importation des bibliothèques nécessaires
import mesa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Configuration de l'affichage des graphiques
%matplotlib inline

# %%
#implementation de l agent
class CitizenAgent(mesa.Agent):
    """Un individu ayant une opinion qui évolue selon ses voisins."""
    def __init__(self, model, alpha):
        super().__init__(model)

        self.opinion = self.model.random.random()
        self.alpha = alpha

    def step(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False
        )

        if len(neighbors) > 0:
            mean_opinion = np.mean(
                [neighbor.opinion for neighbor in neighbors]
            )

            self.opinion = self.opinion + self.alpha * (
                mean_opinion - self.opinion
            )

# %%
#implementation du model
class OpinionModel(mesa.Model):
    """Modèle simulant la formation de l'opinion publique sur une grille."""

    def __init__(self, width, height, alpha=0.2, seed=None):
        super().__init__(seed=seed)

        self.grid = mesa.space.MultiGrid(width, height, torus=True)
        self.alpha = alpha

        for _, (x, y) in self.grid.coord_iter():
            agent = CitizenAgent(self, self.alpha)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Opinion Moyenne": lambda m: np.mean(
                    [a.opinion for a in m.agents]
                ),
                "Variance (Dispersion)": lambda m: np.var(
                    [a.opinion for a in m.agents]
                )
            },
            agent_reporters={
                "Opinion": "opinion"
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

# %%
#execution de la semulation et l'observation 
# Paramètres du modèle
WIDTH = 10
HEIGHT = 10
ALPHA = 0.9  # Modifiez alpha pour observer son impact sur la vitesse de convergence
STEPS = 50

# Création et exécution du modèle
model = OpinionModel(width=WIDTH, height=HEIGHT, alpha=ALPHA, seed=42)

for i in range(STEPS):
    model.step()

# Récupération des données collectées
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()

# Affichage des résultats globaux (Moyenne et Variance)
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Ticks (Temps)')
ax1.set_ylabel('Opinion Moyenne', color=color)
ax1.plot(model_data.index, model_data['Opinion Moyenne'], color=color, label="Moyenne")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Variance (Dispersion)', color=color)
ax2.plot(model_data.index, model_data['Variance (Dispersion)'], color=color, linestyle='--', label="Variance")
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Évolution globale de l'opinion (Moyenne et Dispersion)")
fig.tight_layout()
plt.show()

# Affichage des trajectoires individuelles (Polarisation vs Consensus)
# On prend un échantillon de 20 agents pour la lisibilité
agent_opinions = agent_data.unstack(level=1)['Opinion']
sample_agents = agent_opinions.columns[:20]

plt.figure(figsize=(10, 5))
for agent_id in sample_agents:
    plt.plot(agent_opinions.index, agent_opinions[agent_id], alpha=0.6)

plt.title("Trajectoires des opinions individuelles (Échantillon de 20 agents)")
plt.xlabel("Ticks (Temps)")
plt.ylabel("Opinion [0, 1]")
plt.ylim(0, 1)
plt.show()


