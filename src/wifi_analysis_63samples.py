# step : import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


print("📚 Library siap!")


# STEP 2: GENERATE DATA STRUCTURE UNTUK 63 SAMPLES

# Generate timestamp untuk 63 data (7:30-17:30, setiap 30 menit, 3 data per titik waktu)
times = []
for hour in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
    for minute in [30, 0]:  # 30 menit dan 00 menit
        if hour == 7 and minute == 0:  # Skip jam 7:00
            continue
        if hour == 17 and minute == 30:  # Stop jam 17:30
            break
        for i in range(3):  # 3 data per waktu
            times.append(hour + minute/60)

# Pastikan ada 63 data
times = times[:63]

print(f"🕐 Jumlah timestamp: {len(times)}")
print(f"Range waktu: {min(times):.1f} - {max(times):.1f}")


# STEP 3: LOAD DATA ANDA


# Baca data asli kamu
df = pd.read_csv('dataraw.csv')

print("📊 DATA WIFI (63 samples):")
print(df.head(62))
print(f"\n📈 Shape: {df.shape}")
print(f"🕐 Time range: {df['Time'].min()} - {df['Time'].max()}")

# print(df.columns)

# STEP 4: DATA CLEANING & FEATURE ENGINEERING

# Extract detailed time features
df['Hour'] = df['Time'].astype(int)
df['Minute'] = ((df['Time'] - df['Hour']) * 60).astype(int)
df['Time_Slot'] = df['Hour'] * 100 + df['Minute']  # Format: 730, 800, 830, etc.

# Feature engineering
df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if (8 <= x <= 11) or (14 <= x <= 17) else 0)
df['Is_Morning'] = df['Hour'].apply(lambda x: 1 if 7 <= x <= 10 else 0)
df['Is_Afternoon'] = df['Hour'].apply(lambda x: 1 if 11 <= x <= 17 else 0)
df['Download_Upload_Ratio'] = df['Download'] / df['Upload']
df['Ping_Jitter_Ratio'] = df['Ping'] / df['Jitter']
df['Network_Score'] = (df['Download'] / df['Download'].max() * 0.4 + 
                       (1 - df['Ping'] / df['Ping'].max()) * 0.3 +
                       (1 - df['Jitter'] / df['Jitter'].max()) * 0.3)

# Handle infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# Buat target variable (Kualitas Jaringan)
def categorize_quality(download, ping, jitter):
    if download >= 25 and ping <= 30 and jitter <= 10:
        return 'Baik'
    elif download >= 10 and ping <= 60 and jitter <= 20:
        return 'Sedang'
    else:
        return 'Buruk'

df['Quality'] = df.apply(lambda x: categorize_quality(x['Download'], x['Ping'], x['Jitter']), axis=1)

print("🔧 FEATURE ENGINEERING SELESAI:")
print(f"📊 Distribusi Kualitas:\n{df['Quality'].value_counts()}")
print(f"🕐 Data per jam: {df['Hour'].value_counts().sort_index()}")


# STEP 5: EXPLORATORY DATA ANALYSIS YANG LEBIH DETAIL
# Setup visualisasi
plt.figure(figsize=(20, 15))

# 1. Trend per Time Slot (setiap 30 menit)
plt.subplot(3, 3, 1)
time_slot_avg = df.groupby('Time_Slot').agg({'Download': 'mean', 'Ping': 'mean'})
time_slot_avg['Download'].plot(kind='line', marker='o', color='blue', label='Download')
plt.gca().twinx()
time_slot_avg['Ping'].plot(kind='line', marker='s', color='red', label='Ping', alpha=0.7)
plt.title('Trend Download & Ping per Time Slot')
plt.xticks(rotation=45)

# 2. Distribusi 3 Data per Time Slot
plt.subplot(3, 3, 2)
sample_slot = df[df['Time_Slot'] == 830]  # Contoh untuk jam 8:30
if len(sample_slot) > 0:
    plt.bar(range(len(sample_slot)), sample_slot['Download'], color=['skyblue', 'lightgreen', 'salmon'])
    plt.title(f'Variasi 3 Data di Time Slot 8:30')
    plt.xlabel('Data ke-')
    plt.ylabel('Download Speed')

# 3. Heatmap Kualitas per Jam
plt.subplot(3, 3, 3)
quality_hour_matrix = pd.crosstab(df['Hour'], df['Quality'], normalize='index')
sns.heatmap(quality_hour_matrix, annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Heatmap Kualitas per Jam')

# 4. Boxplot per Jam
plt.subplot(3, 3, 4)
sns.boxplot(data=df, x='Hour', y='Download', palette='Set3')
plt.title('Distribusi Download per Jam')
plt.xticks(rotation=45)

# 5. Violin Plot Ping
plt.subplot(3, 3, 5)
sns.violinplot(data=df, x='Hour', y='Ping', palette='coolwarm')
plt.title('Distribusi Ping per Jam')
plt.xticks(rotation=45)

# 6. Scatter Plot Download vs Ping
plt.subplot(3, 3, 6)
colors = {'Baik': 'green', 'Sedang': 'orange', 'Buruk': 'red'}
plt.scatter(df['Download'], df['Ping'], c=df['Quality'].map(colors), alpha=0.6)
plt.xlabel('Download (Mbps)')
plt.ylabel('Ping (ms)')
plt.title('Download vs Ping (Colored by Quality)')

# 7. Time Series semua metrics
plt.subplot(3, 3, 7)
df_sorted = df.sort_values('Time')
plt.plot(df_sorted['Time'], df_sorted['Download'], label='Download', marker='o')
plt.plot(df_sorted['Time'], df_sorted['Upload'], label='Upload', marker='s')
plt.xlabel('Time')
plt.ylabel('Speed (Mbps)')
plt.title('Time Series Download & Upload')
plt.legend()

# 8. Kualitas per Periode
plt.subplot(3, 3, 8)
period_quality = df.groupby(['Hour', 'Minute'])['Quality'].value_counts().unstack().fillna(0)
period_quality.plot(kind='bar', stacked=True, ax=plt.gca(), color=['red', 'orange', 'green'])
plt.title('Kualitas per Time Point')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("data_analysis_detail.png", dpi=300, bbox_inches='tight')
plt.show()

# Statistical Summary
print("📊 RINGKASAN STATISTIK PER JAM:")
hourly_stats = df.groupby('Hour').agg({
    'Download': ['count', 'mean', 'std', 'min', 'max'],
    'Upload': ['mean', 'std'],
    'Ping': ['mean', 'std'],
    'Jitter': ['mean', 'std'],
    'Quality': lambda x: x.mode()[0]
}).round(2)

print(hourly_stats)

# STEP 6: PREPARE DATA ML DENGAN FEATURE LEBIH BANYAK
# Feature columns yang lebih comprehensive
feature_columns = [
    'Hour', 'Minute', 'Time_Slot', 'Is_Peak_Hour', 'Is_Morning', 'Is_Afternoon',
    'Download', 'Upload', 'Ping', 'Jitter', 
    'Download_Upload_Ratio', 'Ping_Jitter_Ratio', 'Network_Score'
]

X = df[feature_columns]
y = df['Quality']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("🎯 FEATURES UNTUK MODEL:")
print(f"Jumlah features: {len(feature_columns)}")
print(f"Sample features:\n{X.head(3)}")

# Split data dengan stratify untuk menjaga distribusi
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, # stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 DATA SPLIT:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")



# STEP 7: MODEL TRAINING DENGAN CROSS-VALIDATION
from sklearn.model_selection import cross_val_score

# Train model dengan hyperparameter tuning
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

print("🤖 TRAINING MODEL...")
model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"📊 Cross-Validation Scores: {cv_scores}")
print(f"📊 CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 MODEL PERFORMANCE:")
print("="*50)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred)) # target_names=le.classes_))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍  13 FEATURE IMPORTANCE:")
print(feature_importance.head(13))



# STEP 8: ADVANCED VISUALIZATION UNTUK 63 DATA
# Visualisasi khusus untuk data 63 samples
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature Importance
axes[0,0].barh(feature_importance['feature'][:13], feature_importance['importance'][:13])
axes[0,0].set_title('13 Feature Importance')
axes[0,0].set_xlabel('Importance')

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0,1])
axes[0,1].set_title('Confusion Matrix')

# 3. Time-based Quality Prediction
axes[1,0].scatter(df['Time'], df['Download'], c=df['Quality'].map(colors), alpha=0.6, s=60)
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Download Speed')
axes[1,0].set_title('Quality Distribution Over Time')

# 4. Performance by Time Slot
time_slot_performance = df.groupby('Time_Slot').agg({
    'Download': 'mean',
    'Quality': lambda x: (x == 'Baik').mean()  # Proporsi kualitas Baik
})
axes[1,1].plot(time_slot_performance.index, time_slot_performance['Download'], 
               marker='o', label='Download Avg')
axes[1,1].set_xlabel('Time Slot')
axes[1,1].set_ylabel('Download Speed', color='blue')
axes[1,1].tick_params(axis='y', labelcolor='blue')

ax2 = axes[1,1].twinx()
ax2.plot(time_slot_performance.index, time_slot_performance['Quality']*100, 
         marker='s', color='red', label='% Quality Baik')
ax2.set_ylabel('% Quality Baik', color='red')
ax2.tick_params(axis='y', labelcolor='red')
axes[1,1].set_title('Performance & Quality per Time Slot')

plt.tight_layout()
plt.savefig("wifi_analysis_results.png", dpi=300, bbox_inches='tight')
print("✅ Gambar hasil analisis disimpan sebagai: wifi_analysis_results.png")

plt.show()


# STEP 9: DETAILED RECOMMENDATIONS UNTUK SETIAP TIME SLOT
print("📋 REKOMENDASI DETAIL PER TIME SLOT:")
print("="*70)

# Analisis per time slot (setiap 30 menit)
time_slot_analysis = df.groupby(['Hour', 'Minute']).agg({
    'Download': 'mean',
    'Upload': 'mean',
    'Ping': 'mean',
    'Jitter': 'mean',
    'Quality': lambda x: x.mode()[0],
    'Network_Score': 'mean'
}).round(2).reset_index()

# Konversi ke format waktu
time_slot_analysis['Time_Formatted'] = time_slot_analysis.apply(
    lambda x: f"{int(x['Hour'])}:{int(x['Minute']):02d}", axis=1
)
time_slot_analysis.to_csv("time_slot_analysis.csv", index=False, encoding="utf-8")
print("✅ Data analisis per time slot disimpan ke: time_slot_analysis.csv")


recommendation_map = {
    'Baik': '✅ IDEAL: Video conference, streaming 4K, gaming online',
    'Sedang': '⚠️  COCOK: Browsing, YouTube, meeting online', 
    'Buruk': '❌ HINDARI: Gunakan mobile data atau tunggu jam lain'
}


for _, row in time_slot_analysis.iterrows():
    time_str = row['Time_Formatted']
    quality = row['Quality']
    download = row['Download']
    ping = row['Ping']
    score = row['Network_Score']
    
    recommendation = recommendation_map.get(quality, '📊 Analisis data')
    
    print(f"🕐 {time_str:5} | Kualitas: {quality:6} | "
          f"Download: {download:5.1f} Mbps | Ping: {ping:4.1f} ms | "
          f"Score: {score:.2f} | {recommendation}")

# Best and worst time slots
best_slots = time_slot_analysis[time_slot_analysis['Quality'] == 'Baik']['Time_Formatted'].tolist()
worst_slots = time_slot_analysis[time_slot_analysis['Quality'] == 'Buruk']['Time_Formatted'].tolist()

print(f"\n⭐ TIME SLOT TERBAIK: {best_slots}")
print(f"🔻 TIME SLOT TERBURUK: {worst_slots}")

# Additional insights
print(f"\n💡 INSIGHTS UNTUK 63 DATA:")
print(f"- Total samples: {len(df)}")
print(f"- Data per jam: {dict(df['Hour'].value_counts().sort_index())}")
print(f"- Konsistensi: {((df.groupby('Time_Slot')['Quality'].nunique() == 1).mean()*100):.1f}% time slot konsisten")
print(f"- Jam dengan variasi terbesar: {df.groupby('Hour')['Download'].std().idxmax()}")


# STEP 10: SAVE COMPREHENSIVE RESULTS
# Save semua hasil
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model files
joblib.dump(model, f'wifi_63samples_model_{timestamp}.pkl')
joblib.dump(scaler, f'scaler_63samples_{timestamp}.pkl')
joblib.dump(le, f'label_encoder_63samples_{timestamp}.pkl')

# Save analysis results
feature_importance.to_csv(f'feature_importance_63samples_{timestamp}.csv', index=False)
time_slot_analysis.to_csv(f'time_slot_analysis_63samples_{timestamp}.csv', index=False)

# Save predictions
df['Predicted_Quality'] = le.inverse_transform(model.predict(scaler.transform(X)))
df['Prediction_Correct'] = df['Quality'] == df['Predicted_Quality']
df.to_csv(f'complete_analysis_63samples_{timestamp}.csv', index=False)

print("💾 SEMUA HASIL DISIMPAN!")
print(f"✅ Model: wifi_63samples_model_{timestamp}.pkl")
print(f"✅ Analysis: complete_analysis_63samples_{timestamp}.csv")
print(f"✅ Time Slot: time_slot_analysis_63samples_{timestamp}.csv")

# Final comprehensive summary
print(f"\n🎉 ANALISIS 63 DATA SELESAI!")
print("="*60)
print(f"📊 Total Samples: {len(df)}")
print(f"🎯 Model Accuracy: {accuracy:.2%}")
print(f"📈 Cross-Validation: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
print(f"🔧 Features Used: {len(feature_columns)}")
print(f"⭐ Best Time Slots: {len(best_slots)}")
print(f"🔻 Worst Time Slots: {len(worst_slots)}")
print(f"📋 Recommendation Slots: {len(time_slot_analysis)}")