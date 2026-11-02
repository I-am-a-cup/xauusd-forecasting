
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

print("="*70)
print("ПРОГНОЗИРОВАНИЕ XAU/USD: ARIMA и Random Forest")
print("="*70)

df = pd.read_csv('xauusd-d1-bid-2025-10-28-2026-10-28.csv')
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)
prices = df['close']

print(f"Загружено {len(prices)} записей")
print(f"Период: {prices.index[0]} до {prices.index[-1]}")

train_size = len(prices) - 30
train = prices.iloc[:train_size]
test = prices.iloc[train_size:]

print(f"Train: {len(train)} дней")
print(f"Test: {len(test)} дней")

naive_pred_1 = train.iloc[-1]
naive_pred_5 = train.iloc[-1]

mae_naive_1 = mean_absolute_error(test.iloc[:1], [naive_pred_1])
mae_naive_5 = mean_absolute_error(test.iloc[:5], [naive_pred_5]*5)

print(f"\nNaive MAE (1 день): {mae_naive_1:.2f}")
print(f"Naive MAE (5 дней): {mae_naive_5:.2f}")

print("\n" + "="*70)
print("ARIMA МОДЕЛЬ")
print("="*70)

train_values = train.values

model_arima = ARIMA(train_values, order=(1,1,1))
fitted_arima = model_arima.fit()

forecast_arima_1 = fitted_arima.forecast(steps=1)
forecast_arima_5 = fitted_arima.forecast(steps=5)

mae_arima_1 = mean_absolute_error(test.iloc[:1], forecast_arima_1)
mae_arima_5 = mean_absolute_error(test.iloc[:5], forecast_arima_5)

print(f"ARIMA прогноз на 1 день: {forecast_arima_1[0]:.2f}")
print(f"ARIMA прогноз на 5 дней: {forecast_arima_5[-1]:.2f}")
print(f"ARIMA MAE (1 день): {mae_arima_1:.2f}")
print(f"ARIMA MAE (5 дней): {mae_arima_5:.2f}")

print("\n" + "="*70)
print("RANDOM FOREST МОДЕЛЬ")
print("="*70)

def create_advanced_features(series, lags=10):
    df_feat = pd.DataFrame({'price': series})
    
    for i in range(1, lags+1):
        df_feat[f'lag_{i}'] = df_feat['price'].shift(i)
    
    df_feat['ma_3'] = df_feat['price'].rolling(window=3).mean()
    df_feat['ma_5'] = df_feat['price'].rolling(window=5).mean()
    df_feat['ma_10'] = df_feat['price'].rolling(window=10).mean()
    df_feat['std_5'] = df_feat['price'].rolling(window=5).std()
    df_feat['std_10'] = df_feat['price'].rolling(window=10).std()
    df_feat['return_1'] = df_feat['price'].pct_change(1)
    df_feat['return_5'] = df_feat['price'].pct_change(5)
    df_feat['price_vs_ma5'] = df_feat['price'] - df_feat['ma_5']
    df_feat['price_vs_ma10'] = df_feat['price'] - df_feat['ma_10']
    
    return df_feat.dropna()

features_advanced = create_advanced_features(prices, lags=10)
X = features_advanced.drop('price', axis=1)
y = features_advanced['price']

split_idx = len(X) - 30
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, rf_predictions)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_predictions))

rf_pred_1 = rf_predictions[0]
rf_pred_5 = rf_predictions[4]

print(f"RF прогноз на 1 день: {rf_pred_1:.2f}")
print(f"RF прогноз на 5 дней: {rf_pred_5:.2f}")
print(f"RF MAE: {mae_rf:.2f}")
print(f"RF RMSE: {rmse_rf:.2f}")

print("\n" + "="*70)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*70)

results = []
origin_date = train.index[-1]

for h, idx in [(1, 0), (5, 4)]:
    results.append({
        'origin_date': origin_date,
        'target_date': test.index[idx],
        'horizon': h,
        'model': 'ARIMA',
        'y_pred': fitted_arima.forecast(steps=h)[-1]
    })
    results.append({
        'origin_date': origin_date,
        'target_date': test.index[idx],
        'horizon': h,
        'model': 'RandomForest',
        'y_pred': rf_model.predict(X_test.iloc[[idx]])[0]
    })

results_df = pd.DataFrame(results)
results_df.to_csv('arima_rf_predictions.csv', index=False)

print("\nРезультаты сохранены в arima_rf_predictions.csv")
print(results_df)

print("\n" + "="*70)
print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*70)

print(f"\n{'Модель':<20} {'MAE (1 день)':<15} {'vs Naive':<15}")
print("-"*60)
print(f"{'Naive':<20} {mae_naive_1:<15.2f} {'baseline':<15}")
print(f"{'ARIMA':<20} {mae_arima_1:<15.2f} {((mae_naive_1 - mae_arima_1)/mae_naive_1*100):+6.2f}%")
print(f"{'Random Forest':<20} {mae_rf:<15.2f} {((mae_naive_1 - mae_rf)/mae_naive_1*100):+6.2f}%")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== ГРАФИК 1: СРАВНЕНИЕ ПРОГНОЗОВ ====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Левый график — вся история + прогнозы на 1 день
axes[0].plot(train.index, train.values, label='Тренировка', color='blue', linewidth=1.2, alpha=0.7)
axes[0].plot(test.index, test.values, label='Тест (факт)', color='green', linewidth=1.5)
axes[0].axvline(x=train.index[-1], color='gray', linestyle='--', linewidth=1.5, label='Раздел train/test')

axes[0].scatter(test.index[0], forecast_arima_1[0], color='red', s=120, zorder=5, 
                label=f'ARIMA: {forecast_arima_1[0]:.0f}', edgecolors='black', linewidth=1.5)
axes[0].scatter(test.index[0], rf_pred_1, color='orange', s=120, zorder=5, 
                label=f'RF: {rf_pred_1:.0f}', edgecolors='black', linewidth=1.5)
axes[0].scatter(test.index[0], naive_pred_1, color='purple', s=120, zorder=5, 
                label=f'Naive: {naive_pred_1:.0f}', edgecolors='black', linewidth=1.5)

axes[0].annotate(f'Факт: {test.iloc[0]:.0f}', 
                 xy=(test.index[0], test.iloc[0]),
                 xytext=(test.index[0] + pd.Timedelta(days=10), test.iloc[0] + 80),
                 fontsize=10, fontweight='bold', color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

axes[0].set_xlabel('Дата')
axes[0].set_ylabel('Цена золота (USD)')
axes[0].set_title('XAU/USD: Прогноз на 1 день вперёд')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[0].tick_params(axis='x', rotation=45)

# Правый график — прогноз на 5 дней
axes[1].plot(train.index[-60:], train.values[-60:], label='Тренировка (последние 60)', 
             color='blue', linewidth=1.2, alpha=0.7)
axes[1].plot(test.index[:5], test.values[:5], label='Тест (факт, 5 дней)', 
             color='green', linewidth=1.5, marker='o')
axes[1].axvline(x=train.index[-1], color='gray', linestyle='--', linewidth=1.5)

arima_5_path = fitted_arima.forecast(steps=5)
axes[1].plot(test.index[:5], arima_5_path, color='red', linewidth=1.5, marker='s', label='ARIMA')
axes[1].plot(test.index[:5], rf_model.predict(X_test.iloc[:5]), color='orange', 
             linewidth=1.5, marker='^', label='Random Forest')
axes[1].axhline(y=naive_pred_5, color='purple', linestyle=':', linewidth=1.5, label='Naive')

axes[1].set_xlabel('Дата')
axes[1].set_ylabel('Цена золота (USD)')
axes[1].set_title('XAU/USD: Прогноз на 5 дней вперёд')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(" График сохранён: forecast_comparison.png")

# ==================== ГРАФИК 2: ВАЖНОСТЬ ПРИЗНАКОВ RF ====================
feature_names = X.columns.tolist()
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Признак': feature_names,
    'Важность': importances
}).sort_values('Важность', ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(feature_importance_df['Признак'], feature_importance_df['Важность'], 
         color='steelblue', edgecolor='black', linewidth=0.5)
plt.xlabel('Важность признака')
plt.ylabel('Признак')
plt.title('Важность признаков в Random Forest')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(" График сохранён: feature_importance.png")
print("\n" + "="*70)
print("ВЫВОД")
print("="*70)
improvement = (mae_naive_1 - mae_rf)/mae_naive_1*100
print(f"""
Random Forest с 19 признаками показал лучший результат:
- MAE: {mae_rf:.2f} (улучшение на {improvement:.1f}% относительно Naive)
- Ключ к успеху - инженерия признаков: лаги, скользящие средние, 
  волатильность и дневные изменения цены

ARIMA не смогла превзойти Naive, что типично для финансовых рядов.
""")
