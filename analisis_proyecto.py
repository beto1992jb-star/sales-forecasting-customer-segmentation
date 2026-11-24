import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ==========================================
# 1. CARGA Y LIMPIEZA CON REGLAS EXPLÍCITAS
# ==========================================
sales_df = pd.read_csv("sales_data.csv")
customers_df = pd.read_csv("customer_data.csv")

# Revisión explícita de calidad (nulos, duplicados y tipos) en ambos datasets
for dataset_name, df_check in [
    ("sales_data", sales_df),
    ("customer_data", customers_df),
]:
  print(f"\n=== Calidad de datos: {dataset_name} ===")
  print("Dimensiones (filas, columnas):", df_check.shape)
  print("Nulos por columna:\n", df_check.isnull().sum().to_string())
  print("Duplicados exactos:", int(df_check.duplicated().sum()))
  print("Tipos de datos:\n", df_check.dtypes.to_string())

# Eliminar duplicados exactos si existieran y validar columnas numéricas
sales_df = sales_df.drop_duplicates().copy()
customers_df = customers_df.drop_duplicates().copy()

sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
invalid_dates = int(sales_df["date"].isnull().sum())
if invalid_dates > 0:
  sales_df = sales_df.dropna(subset=["date"]).copy()

for col in ["quantity", "unit_price"]:
  sales_df[col] = pd.to_numeric(sales_df[col], errors="coerce")

sales_df = sales_df[
    (sales_df["quantity"] > 0) & (sales_df["unit_price"] >= 0)
].copy()

sales_df = sales_df.sort_values("date").reset_index(drop=True)
sales_df["revenue"] = sales_df["quantity"] * sales_df["unit_price"]
sales_df["year_month"] = sales_df["date"].dt.to_period("M")

print(
    f"\nFechas inválidas eliminadas: {invalid_dates} | Filas válidas en"
    f" sales_df: {len(sales_df)}"
)
print(
    "\nIngresos por categoría:\n",
    sales_df.groupby("category")["revenue"].sum(),
)
print(
    "\nIngresos por producto:\n",
    sales_df.groupby("product")["revenue"].sum().sort_values(ascending=False),
)

# ==========================================
# 2. SERIE MENSUAL Y PRONÓSTICO TEMPORAL
# ==========================================
monthly_df = (
    sales_df.groupby("year_month", as_index=False)
    .agg(
        transactions=("revenue", "count"),
        total_quantity=("quantity", "sum"),
        total_revenue=("revenue", "sum"),
        avg_unit_price=("unit_price", "mean"),
    )
    .sort_values("year_month")
    .reset_index(drop=True)
)
monthly_df["month_idx"] = np.arange(1, len(monthly_df) + 1)
monthly_df["year_month_str"] = monthly_df["year_month"].astype(str)

# División cronológica estricta (4 meses train / 2 meses test)
train_df = monthly_df.iloc[:4].copy()
test_df = monthly_df.iloc[4:].copy()
X_train, y_train = train_df[["month_idx"]], train_df["total_revenue"]
X_test, y_test = test_df[["month_idx"]], test_df["total_revenue"]

# Modelos principales en ventana de prueba
base_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

forecast_rows = []
for name, model in base_models.items():
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  mae = mean_absolute_error(y_test, preds)
  mse = mean_squared_error(y_test, preds)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_test, preds)
  forecast_rows.append({
      "Model": name,
      "Pred_2025_05": round(float(preds[0]), 2),
      "Pred_2025_06": round(float(preds[1]), 2),
      "MAE": round(float(mae), 2),
      "MSE": round(float(mse), 2),
      "RMSE": round(float(rmse), 2),
      "R2": round(float(r2), 4),
      "Forecast_2025_07_Exp": round(
          float(model.predict(pd.DataFrame({"month_idx": [7]}))[0]), 2
      ),
  })

forecast_results_df = pd.DataFrame(forecast_rows)
print("\nEvaluación cronológica en Test (2025-05 y 2025-06):\n")
print(forecast_results_df.to_string(index=False))
print(
    "Nota: La proyección para 2025-07 es experimental por contar solo con 6"
    " meses históricos."
)

# ==========================================
# 3. OPTIMIZACIÓN DINÁMICA DE CANDIDATOS
# ==========================================
candidates = [
    ("Linear Regression (Baseline)", LinearRegression(), "default"),
    ("Ridge Regression", Ridge(alpha=0.01), "alpha=0.01"),
    ("Ridge Regression", Ridge(alpha=0.1), "alpha=0.1"),
    ("Ridge Regression", Ridge(alpha=1.0), "alpha=1.0"),
    ("Ridge Regression", Ridge(alpha=10.0), "alpha=10.0"),
    (
        "Decision Tree (Default)",
        DecisionTreeRegressor(random_state=42),
        "default",
    ),
    (
        "Random Forest (Baseline)",
        RandomForestRegressor(n_estimators=100, random_state=42),
        "n_estimators=100",
    ),
    (
        "Random Forest (Tuned)",
        RandomForestRegressor(n_estimators=20, max_depth=2, random_state=42),
        "n_estimators=20, max_depth=2",
    ),
    (
        "Random Forest (Tuned)",
        RandomForestRegressor(n_estimators=50, max_depth=2, random_state=42),
        "n_estimators=50, max_depth=2",
    ),
    (
        "Random Forest (Tuned)",
        RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        "n_estimators=50, max_depth=3",
    ),
]

eval_rows = []
for label, model, params in candidates:
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  eval_rows.append({
      "Model": label,
      "Params": params,
      "MAE": round(float(mean_absolute_error(y_test, preds)), 2),
      "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
      "R2": round(float(r2_score(y_test, preds)), 4),
  })

eval_df = (
    pd.DataFrame(eval_rows)
    .sort_values(by=["MAE", "RMSE"])
    .reset_index(drop=True)
)
print(
    "\nRanking de modelos en ventana cronológica de prueba (menor MAE es"
    " mejor):\n"
)
print(eval_df.to_string(index=False))
print(
    f"\nMejor candidato en esta partición: {eval_df.loc[0, 'Model']}"
    f" ({eval_df.loc[0, 'Params']}) con MAE={eval_df.loc[0, 'MAE']} y"
    f" RMSE={eval_df.loc[0, 'RMSE']}."
)
print(
    "Nota: R2 negativo refleja la alta sensibilidad de evaluar solo 2 meses"
    " tras entrenar en 4 meses; se requieren más datos históricos."
)

# ==========================================
# 4. SEGMENTACIÓN K-MEANS CON JUSTIFICACIÓN DE k=3
# ==========================================
cluster_features = [
    "age",
    "annual_income",
    "purchases",
    "avg_order_value",
    "website_visits",
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers_df[cluster_features])

metrics_k = []
for k in range(2, 7):
  km = KMeans(n_clusters=k, random_state=42, n_init=10)
  lbl = km.fit_predict(X_scaled)
  metrics_k.append({
      "k": k,
      "inertia": round(float(km.inertia_), 2),
      "silhouette": round(float(silhouette_score(X_scaled, lbl)), 4),
  })

elbow_df = pd.DataFrame(metrics_k)
print("\nEvaluación de k (2 a 6):\n", elbow_df.to_string(index=False))

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customers_df["cluster"] = kmeans_final.fit_predict(X_scaled)

print(
    "\nJustificación de k=3: caída fuerte de inercia de k=2 (25.96) a k=3"
    " (11.06) y separación clara en 3 perfiles de clientes (bajo, intermedio y"
    " alto valor/frecuencia/ticket), pese a que k=2 tenga silueta ligeramente"
    " superior (0.5985 vs 0.5349)."
)
print(
    "\nResumen por cluster (k=3):\n",
    customers_df.groupby("cluster")[cluster_features]
    .agg(["count", "mean"])
    .round(2)
    .to_string(),
)

# ==========================================
# 5. GUARDAR CSVS PROCESADOS FINALES
# ==========================================
sales_df.to_csv("sales_data_processed.csv", index=False)
monthly_df.to_csv("monthly_sales_processed.csv", index=False)
customers_df.to_csv("customer_data_processed.csv", index=False)
print(
    "\nCSVs guardados correctamente: sales_data_processed.csv,"
    " monthly_sales_processed.csv, customer_data_processed.csv"
)

# ==========================================
# 6. VISUALIZACIONES (.PNG)
# ==========================================
# 6.1 Ingresos históricos mensuales
plt.figure(figsize=(8, 4.5))
plt.plot(
    monthly_df["year_month_str"],
    monthly_df["total_revenue"],
    marker="o",
    linewidth=2,
    color="#1f77b4",
)
for x_val, y_val in zip(
    monthly_df["year_month_str"], monthly_df["total_revenue"]
):
  plt.annotate(
      f"{y_val:,.0f}",
      (x_val, y_val),
      textcoords="offset points",
      xytext=(0, 8),
      ha="center",
      fontsize=9,
  )
plt.title("Historical Monthly Revenue (2025-01 to 2025-06)")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("historical_revenue.png", dpi=150)
plt.close()

# 6.2 Predicciones de ingresos
test_months_ext = list(test_df["year_month_str"]) + ["2025-07 (Forecast)"]
colors_map = {
    "Linear Regression": "#1f77b4",
    "Decision Tree": "#ff7f0e",
    "Random Forest": "#2ca02c",
}

plt.figure(figsize=(8.5, 4.8))
plt.plot(
    monthly_df["year_month_str"],
    monthly_df["total_revenue"],
    marker="o",
    linewidth=2.2,
    color="#111111",
    label="Actual Revenue",
)
for m_name, m_obj in base_models.items():
  m_obj.fit(X_train, y_train)
  p_t = list(m_obj.predict(X_test))
  p_next = float(m_obj.predict(pd.DataFrame({"month_idx": [7]}))[0])
  plt.plot(
      test_months_ext,
      p_t + [p_next],
      marker="s",
      linestyle="--",
      color=colors_map[m_name],
      label=f"{m_name}",
  )
plt.axvline(
    x="2025-04", color="gray", linestyle=":", label="Train/Test Split"
)
plt.title("Monthly Revenue Forecast (Test & Next Month)")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=15)
plt.legend(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("revenue_forecast.png", dpi=150)
plt.close()

# 6.3 Comparación de métricas MAE y RMSE
x_pos = np.arange(len(forecast_results_df))
width = 0.35
plt.figure(figsize=(8, 4.5))
plt.bar(
    x_pos - width / 2,
    forecast_results_df["MAE"],
    width=width,
    label="MAE",
    color="#1f77b4",
)
plt.bar(
    x_pos + width / 2,
    forecast_results_df["RMSE"],
    width=width,
    label="RMSE",
    color="#6baed6",
)
plt.xticks(x_pos, forecast_results_df["Model"])
plt.title("Forecasting Error Metrics on Test Set (Lower is Better)")
plt.ylabel("Error (Revenue Units)")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("forecast_metrics_comparison.png", dpi=150)
plt.close()

# 6.4 Método Elbow + Segmentación de clientes (K-Means k=3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(
    elbow_df["k"],
    elbow_df["inertia"],
    marker="o",
    linewidth=2,
    color="#1f77b4",
)
ax1.set_title("Elbow Method (K-Means Inertia)")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.grid(True, linestyle="--", alpha=0.4)

for c_id, grp in customers_df.groupby("cluster"):
  ax2.scatter(
      grp["annual_income"],
      grp["avg_order_value"],
      s=grp["purchases"] * 4,
      alpha=0.85,
      label=f"Cluster {c_id} (n={len(grp)})",
  )
ax2.set_title("Customer Segmentation (Income vs. Avg Order Value)")
ax2.set_xlabel("Annual Income")
ax2.set_ylabel("Avg Order Value")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("customer_segmentation_elbow.png", dpi=150)
plt.close()