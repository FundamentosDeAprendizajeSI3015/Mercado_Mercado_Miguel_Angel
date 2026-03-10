import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import uniform, randint
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Crear directorio de outputs - usando ruta absoluta basada en el directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'outputs')
os.makedirs(output_dir, exist_ok=True)
print(f"Directorio de outputs: {output_dir}\n")

print("=" * 80)
print("1. DESCARGANDO Y CARGANDO EL DATASET")
print("=" * 80)

# Descargar dataset (kagglehub debe estar configurado en el entorno)
path = kagglehub.dataset_download("bharatnatrayn/movies-dataset-for-feature-extracion-prediction")
print(f"Path to dataset files: {path}\n")

# Determinar directorio de datos (si es zip, extraer)
data_dir = None
if os.path.isfile(path):
	if path.lower().endswith('.zip'):
		import zipfile
		extract_dir = os.path.join(output_dir, 'dataset_extract')
		os.makedirs(extract_dir, exist_ok=True)
		with zipfile.ZipFile(path, 'r') as z:
			z.extractall(extract_dir)
		data_dir = extract_dir
	else:
		data_dir = os.path.dirname(path) or os.getcwd()
elif os.path.isdir(path):
	data_dir = path
else:
	data_dir = os.path.dirname(path) or os.getcwd()

print(f"Data directory: {data_dir}\n")

# Buscar CSVs
csv_files = []
for root, _, files in os.walk(data_dir):
	for f in files:
		if f.lower().endswith('.csv'):
			csv_files.append(os.path.join(root, f))

if not csv_files:
	raise FileNotFoundError("No se encontraron archivos CSV en el dataset descargado.")

csv_path = csv_files[0]
print(f"Usando CSV: {csv_path}\n")

# Cargar CSV
try:
	df = pd.read_csv(csv_path, low_memory=False)
except Exception:
	df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)

print("Primeras filas:\n", df.head(3))
print('\nShape:', df.shape)

# Exploración rápida
info_path = os.path.join(output_dir, 'dataset_quickinfo.txt')
with open(info_path, 'w', encoding='utf-8') as f:
	f.write(str(df.info()) + '\n\n')
	f.write('Nulos por columna:\n')
	f.write(str(df.isnull().sum()) + '\n')
	f.write('\nDescripción numérica:\n')
	f.write(str(df.describe(include='all')))

print(f"Resumen rápido guardado en: {info_path}\n")

# Limpieza/transformación básica
# 1) Eliminar duplicados
df = df.drop_duplicates()

# 2) Detectar columna objetivo o crear una variable binaria 'target'
# Detectar columna objetivo preferida (insensible a mayúsculas)
target_col = None
cols_lower = {c.lower(): c for c in df.columns}
if 'rating' in cols_lower:
	target_col = cols_lower['rating']
elif 'votes' in cols_lower:
	target_col = cols_lower['votes']
elif 'genre' in cols_lower:
	target_col = cols_lower['genre']
else:
	# buscar nombres numéricos relevantes
	for c in df.select_dtypes(include=[np.number]).columns:
		if any(k in c.lower() for k in ['revenue', 'profit', 'score']):
			target_col = c
			break

if target_col is None:
	raise ValueError('No se encontró una columna objetivo evidente. Por favor indique manualmente cuál usar.')

print(f"Columna objetivo detectada: {target_col}")

# Si la columna objetivo está en formato string que representa números, convertirla
if df[target_col].dtype == object:
	df[target_col] = df[target_col].astype(str).str.replace('[^0-9.]', '', regex=True)
	df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# Si la columna objetivo es continua (numérica), convertir a binaria
if pd.api.types.is_numeric_dtype(df[target_col]):
	if target_col.lower() == 'rating' or 'rating' in target_col.lower():
		# umbral sugerido para rating
		thresh = 7.0
		# si no hay suficientes datos por encima, usar la mediana
		if df[target_col].dropna().max() <= thresh:
			thresh = df[target_col].median()
	else:
		thresh = df[target_col].median()
	df['target'] = (df[target_col] > thresh).astype(int)
	print(f"Se creó columna binaria 'target' con umbral={thresh}")
else:
	# Si es categórica, mapear a números
	df['target'] = pd.factorize(df[target_col].astype(str))[0]
	print("Se creó columna 'target' codificando categorías")

# 3) Eliminar columnas irrelevantes: identificadores y texto largo
drop_cols = []
text_cols = df.select_dtypes(include=['object']).columns.tolist()
for c in text_cols:
	# columnas de texto muy largas (descripciones) o con alta cardinalidad
	if df[c].nunique(dropna=False) > 0.8 * len(df) or df[c].str.len().dropna().mean() > 100:
		drop_cols.append(c)

# columnas con demasiados nulos
for c in df.columns:
	if df[c].isnull().mean() > 0.6:
		drop_cols.append(c)

# columnas obviamente identificadoras
for k in ['id', 'movie_id', 'imdb_id', 'title', 'homepage', 'overview', 'tagline', 'poster_path']:
	if k in df.columns:
		drop_cols.append(k)

drop_cols = list(set(drop_cols))
print(f"Columnas a eliminar por limpieza: {drop_cols}\n")
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Preparar features X y target y
y = df['target']
X = df.drop(columns=['target', target_col], errors='ignore')

# Seleccionar columnas numéricas y categóricas
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Features numéricas: {len(numeric_cols)}, categóricas: {len(categorical_cols)}")

# Definir preprocesador
numeric_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
	('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
	('num', numeric_transformer, numeric_cols),
	('cat', categorical_transformer, categorical_cols)
], remainder='drop')

# Dividir dataset (estratificar por target binario)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Split realizado. Train: {X_train.shape}, Test: {X_test.shape}\n")

# Función auxiliar para entrenar y evaluar
def train_and_evaluate(model, model_name, param_distributions=None):
	pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', model)])
	if param_distributions is not None:
		search = RandomizedSearchCV(pipe, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=42)
		search.fit(X_train, y_train)
		best = search.best_estimator_
		print(f"{model_name} mejor params: {search.best_params_}")
	else:
		best = pipe
		best.fit(X_train, y_train)

	preds = best.predict(X_test)
	acc = accuracy_score(y_test, preds)
	prec = precision_score(y_test, preds, zero_division=0)
	rec = recall_score(y_test, preds, zero_division=0)
	f1 = f1_score(y_test, preds, zero_division=0)
	cm = confusion_matrix(y_test, preds)

	# Guardar matriz de confusión
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	fig, ax = plt.subplots(figsize=(5,4))
	disp.plot(ax=ax)
	fig.suptitle(f"Matriz de confusión - {model_name}")
	cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
	fig.savefig(cm_path, bbox_inches='tight')
	plt.close(fig)

	# Importancia de características si disponible
	fi_path = None
	try:
		clf = best.named_steps['clf']
		if hasattr(clf, 'feature_importances_'):
			# obtener nombres de columnas luego del preprocesador
			X_sample = X_train.head(10)
			X_trans = preprocessor.fit_transform(X_sample)
			# construir nombres para features one-hot
			ohe_cols = []
			if categorical_cols:
				ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
				cat_names = ohe.get_feature_names_out(categorical_cols)
				ohe_cols = list(cat_names)
			feature_names = list(numeric_cols) + ohe_cols
			importances = clf.feature_importances_
			# tomar top 20
			idx = np.argsort(importances)[-20:][::-1]
			fig2, ax2 = plt.subplots(figsize=(8,6))
			ax2.barh(np.array(feature_names)[idx], importances[idx])
			ax2.set_title(f'Feature importances - {model_name}')
			plt.tight_layout()
			fi_path = os.path.join(output_dir, f'feature_importances_{model_name}.png')
			fig2.savefig(fi_path, bbox_inches='tight')
			plt.close(fig2)
	except Exception:
		fi_path = None

	# Guardar modelo
	model_file = os.path.join(output_dir, f'{model_name}_model.joblib')
	joblib.dump(best, model_file)

	metrics = {
		'model': model_name,
		'accuracy': acc,
		'precision': prec,
		'recall': rec,
		'f1': f1,
		'confusion_matrix_image': cm_path,
		'feature_importance_image': fi_path,
		'model_file': model_file
	}
	return metrics

# Definir modelos y espacios de búsqueda sencillos
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf_params = {
	'clf__n_estimators': randint(50, 200),
	'clf__max_depth': randint(3, 20)
}

gb_params = {
	'clf__n_estimators': randint(50, 200),
	'clf__learning_rate': uniform(0.01, 0.3),
	'clf__max_depth': randint(3, 10)
}

print("Entrenando Random Forest...")
rf_metrics = train_and_evaluate(rf, 'random_forest', rf_params)
print("Entrenando Gradient Boosting...")
gb_metrics = train_and_evaluate(gb, 'gradient_boosting', gb_params)

# Guardar métricas comparativas
metrics_df = pd.DataFrame([rf_metrics, gb_metrics])
metrics_csv = os.path.join(output_dir, 'metrics_comparison.csv')
metrics_df.to_csv(metrics_csv, index=False)
print(f"Métricas guardadas en: {metrics_csv}\n")

print("Procesamiento completado. Revisa la carpeta outputs para resultados y modelos.")
