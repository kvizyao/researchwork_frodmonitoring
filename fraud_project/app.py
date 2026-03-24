from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import threading
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fraud-monitoring-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

class TransactionGenerator:
    def __init__(self):
        self.is_running = False
        self.transaction_count = 0
        self.stats = {'total': 0, 'normal': 0, 'fraud': 0, 'rule_blocked': 0, 'ml_blocked': 0}
        self.normal_users = [f"USER_{i:04d}" for i in range(100, 500)]
        self.fraud_users = [f"FRAUD_{i:03d}" for i in range(1, 21)]
        self.countries = ['RU', 'US', 'DE', 'CN', 'BR', 'FR', 'NL', 'TR', 'UA', 'KZ']
        self.devices = ['iPhone', 'Android', 'Windows', 'Mac', 'Linux']
        self.merchants = ['Яндекс.Маркет', 'OZON', 'Wildberries', 'AliExpress Россия', 'СберМаркет', 'DNS', 'М.Видео', 'Эльдорадо', 'Steam', 'AppStore']
        self.ml_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
        self.ml_trained = False
        self.transactions_history = []

    def generate_transaction(self):
        self.transaction_count += 1
        is_fraud = random.random() < 0.15
        if is_fraud:
            user = random.choice(self.fraud_users)
            amount = np.random.lognormal(mean=11.0, sigma=0.5) * random.uniform(1.2, 3.0)
            country = random.choice(['BR', 'TR', 'NG', 'CN', 'RU'])
            merchant = random.choice(['Steam', 'AppStore', 'Amazon', 'PlayStation', 'Стим', 'Epic Games'])
            device = random.choice(['Android', 'Windows'])
            ip_changed = random.random() < 0.6
            first_purchase = random.random() < 0.5
            hour = random.randint(0, 6)
        else:
            user = random.choice(self.normal_users)
            amount = np.random.lognormal(mean=8.5, sigma=0.8) * random.uniform(0.8, 1.5)
            country = random.choice(self.countries)
            merchant = random.choice(self.merchants)
            device = random.choice(self.devices)
            ip_changed = random.random() < 0.1
            first_purchase = random.random() < 0.05
            hour = random.randint(9, 20)
        amount = int(amount)
        transaction = {
            'id': f"TX{self.transaction_count:07d}{'F' if is_fraud else 'N'}",
            'user': user,
            'amount': amount,
            'currency': '₽',
            'country': country,
            'merchant': merchant,
            'device': device,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ip_changed': ip_changed,
            'first_purchase': first_purchase,
            'is_fraud': is_fraud,
            'hour': hour,
            'type': 'МОШЕННИЧЕСТВО' if is_fraud else 'НОРМАЛЬНАЯ'
        }
        return transaction

    def analyze_with_rules(self, transaction):
        score = 0
        triggered_rules = []
        if transaction['amount'] > 100000:
            score += 40
            triggered_rules.append('Сумма > 100,000 ₽')
        if transaction['country'] in ['BR', 'TR', 'NG', 'CN']:
            score += 25
            triggered_rules.append('Высокорисковая страна')
        if transaction['hour'] < 6:
            score += 20
            triggered_rules.append('Ночное время (00:00-06:00)')
        if transaction['ip_changed']:
            score += 15
            triggered_rules.append('Смена IP')
        if transaction['first_purchase']:
            score += 25
            triggered_rules.append('Первая покупка')
        if transaction['merchant'] in ['Steam', 'AppStore', 'PlayStation', 'Стим', 'Epic Games']:
            score += 10
            triggered_rules.append('Цифровой товар')
        is_fraud = score >= 50
        return {'score': score, 'rules_triggered': triggered_rules, 'decision': 'БЛОКИРОВКА' if is_fraud else 'РАЗРЕШЕНО', 'is_fraud': is_fraud}

    def analyze_with_ml(self, transaction):
        if not self.ml_trained:
            return {'decision': 'НЕ ОБУЧЕНА', 'confidence': 0.0, 'is_fraud': False}
        try:
            features = pd.DataFrame([{
                'amount': transaction['amount'],
                'hour': transaction['hour'],
                'ip_changed': int(transaction['ip_changed']),
                'first_purchase': int(transaction['first_purchase']),
                'high_risk_country': int(transaction['country'] in ['BR', 'TR', 'NG', 'CN']),
                'digital_goods': int(transaction['merchant'] in ['Steam', 'AppStore', 'PlayStation', 'Стим', 'Epic Games'])
            }])
            prediction = self.ml_model.predict(features)[0]
            probabilities = self.ml_model.predict_proba(features)[0]
            confidence = float(probabilities[1] if prediction == 1 else probabilities[0])

            return {'decision': 'БЛОКИРОВКА' if prediction == 1 else 'РАЗРЕШЕНО', 'confidence': confidence, 'is_fraud': bool(prediction == 1)}
        except Exception as e:
            print(f"ML ошибка: {e}")
            return {'decision': 'ОШИБКА', 'confidence': 0.0, 'is_fraud': False}

    def train_ml_model(self):
        if len(self.transactions_history) < 50:
            return False, "Недостаточно данных для обучения (нужно минимум 50 транзакций)"
        try:
            X = pd.DataFrame([{
                'amount': tx['amount'],
                'hour': tx['hour'],
                'ip_changed': int(tx['ip_changed']),
                'first_purchase': int(tx['first_purchase']),
                'high_risk_country': int(tx['country'] in ['BR', 'TR', 'NG', 'CN']),
                'digital_goods': int(tx['merchant'] in ['Steam', 'AppStore', 'PlayStation', 'Стим', 'Epic Games'])
            } for tx in self.transactions_history])
            y = [tx['is_fraud'] for tx in self.transactions_history]
            if len(set(y)) < 2:
                return False, "Нужны транзакции обоих типов (нормальные и мошеннические)"
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            self.ml_model.fit(X_train, y_train)
            from sklearn.metrics import accuracy_score
            y_pred = self.ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.ml_trained = True
            return True, f"Модель обучена! Точность: {accuracy:.1%}"
        except Exception as e:
            return False, f"Ошибка обучения: {str(e)}"

    def start_generation(self, speed=2.0):
        self.is_running = True
        def generation_loop():
            while self.is_running:
                transaction = self.generate_transaction()
                rule_result = self.analyze_with_rules(transaction)
                ml_result = self.analyze_with_ml(transaction)
                self.stats['total'] += 1
                if transaction['is_fraud']:
                    self.stats['fraud'] += 1
                else:
                    self.stats['normal'] += 1

                if rule_result['is_fraud']:
                    self.stats['rule_blocked'] += 1
                if ml_result['is_fraud']:
                    self.stats['ml_blocked'] += 1
                tx_record = {
                    **transaction,
                    'rule_score': rule_result['score'],
                    'rule_decision': rule_result['decision'],
                    'ml_decision': ml_result['decision'],
                    'ml_confidence': ml_result['confidence']
                }
                self.transactions_history.append(tx_record)
                socket_data = {
                    'transaction': {
                        'id': transaction['id'],
                        'user': transaction['user'],
                        'amount': float(transaction['amount']),
                        'currency': transaction['currency'],
                        'country': transaction['country'],
                        'merchant': transaction['merchant'],
                        'device': transaction['device'],
                        'timestamp': transaction['timestamp'],
                        'type': transaction['type'],
                        'is_fraud': bool(transaction['is_fraud'])
                    },
                    'rule_result': {
                        'score': int(rule_result['score']),
                        'rules_triggered': rule_result['rules_triggered'],
                        'decision': rule_result['decision'],
                        'is_fraud': bool(rule_result['is_fraud'])
                    },
                    'ml_result': {
                        'decision': ml_result['decision'],
                        'confidence': float(ml_result['confidence']),
                        'is_fraud': bool(ml_result['is_fraud'])
                    },
                    'stats': {
                        'total': int(self.stats['total']),
                        'normal': int(self.stats['normal']),
                        'fraud': int(self.stats['fraud']),
                        'rule_blocked': int(self.stats['rule_blocked']),
                        'ml_blocked': int(self.stats['ml_blocked']),
                        'fraud_rate': float(self.stats['fraud'] / self.stats['total'] if self.stats['total'] > 0 else 0)
                    }
                }
                socketio.emit('new_transaction', socket_data)
                time.sleep(1.0 / speed)
        thread = threading.Thread(target=generation_loop, daemon=True)
        thread.start()
        return thread

    def stop_generation(self):
        self.is_running = False

    def get_chart_data(self):
        try:
            # Устанавливаем глобальные параметры шрифта для matplotlib
            plt.rcParams.update({
                'font.size': 15,  # Основной размер шрифта
                'axes.labelsize': 15,  # Размер подписей осей
                'axes.titlesize': 16,  # Размер заголовков
                'xtick.labelsize': 15,  # Размер подписей делений по X
                'ytick.labelsize': 15,  # Размер подписей делений по Y
                'legend.fontsize': 15,  # Размер шрифта легенды
                'figure.titlesize': 16  # Размер заголовка фигуры
            })

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            if not self.transactions_history or self.stats['total'] == 0:
                for ax, title in zip([ax1, ax2, ax3, ax4],
                                     ['Распределение транзакций', 'Эффективность методов блокировки',
                                      'Суммы транзакций', 'Активность по часам']):
                    ax.text(0.5, 0.5, 'Данные появятся после запуска мониторинга',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=14, color='gray')
                    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                    ax.set_facecolor('#f8f9fa')
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                # График 1: Круговая диаграмма
                labels = ['Нормальные', 'Мошеннические']
                sizes = [self.stats['normal'], self.stats['fraud']]
                colors = ['#2ecc71', '#e74c3c']
                if sum(sizes) > 0:
                    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                                       autopct='%1.1f%%', startangle=90,
                                                       textprops={'fontsize': 15})
                    # Увеличиваем шрифт для подписей и процентов
                    for text in texts:
                        text.set_fontsize(15)
                    for autotext in autotexts:
                        autotext.set_fontsize(15)
                        autotext.set_fontweight('bold')
                    ax1.set_title('Распределение транзакций', fontsize=16, pad=20)
                else:
                    ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=15)

                # График 2: Эффективность методов блокировки
                methods = ['Правила', 'MO-модель']
                blocked = [self.stats['rule_blocked'], self.stats['ml_blocked']]
                colors = ['#3498db', '#9b59b6']
                bars = ax2.bar(methods, blocked, color=colors, width=0.6)
                ax2.set_title('Эффективность методов блокировки', fontsize=16, pad=20)
                ax2.set_ylabel('Количество блокировок', fontsize=15)
                ax2.tick_params(axis='x', labelsize=15)
                ax2.tick_params(axis='y', labelsize=15)

                # Добавляем значения над столбцами
                for bar, value in zip(bars, blocked):
                    if value > 0:
                        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                                 str(value), ha='center', va='bottom', fontsize=15)

                # График 3: Суммы последних транзакций
                recent_tx = self.transactions_history[-15:] if len(
                    self.transactions_history) > 15 else self.transactions_history
                if recent_tx:
                    amounts = [tx['amount'] for tx in recent_tx]
                    colors = ['red' if tx['is_fraud'] else 'green' for tx in recent_tx]
                    bars = ax3.bar(range(len(amounts)), amounts, color=colors, alpha=0.8, width=0.7)
                    ax3.set_title('Суммы последних транзакций', fontsize=16, pad=20)
                    ax3.set_xlabel('Номер транзакции', fontsize=15)
                    ax3.set_ylabel('Сумма (₽)', fontsize=15)
                    ax3.tick_params(axis='x', labelsize=15)
                    ax3.tick_params(axis='y', labelsize=15)
                    ax3.set_xticks(range(len(amounts)))
                    ax3.set_xticklabels([tx['id'][-4:] for tx in recent_tx], rotation=45, fontsize=10)

                    # Добавляем подписи для крупных сумм
                    for bar, amount in zip(bars, amounts):
                        if amount > 50000:
                            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                                     f'{amount:,}'.replace(',', ' '), ha='center', va='bottom',
                                     fontsize=15)
                else:
                    ax3.text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=15)

                # График 4: Активность по часам
                if self.transactions_history:
                    hours = [tx['hour'] for tx in self.transactions_history]
                    fraud_hours = [tx['hour'] for tx in self.transactions_history if tx['is_fraud']]

                    ax4.hist([hours, fraud_hours], bins=24, range=(0, 24),
                             label=['Все транзакции', 'Мошеннические'],
                             color=['blue', 'red'], alpha=0.7)
                    ax4.set_title('Активность транзакций по часам', fontsize=16, pad=20)
                    ax4.set_ylabel('Количество транзакций', fontsize=15)
                    ax4.legend(fontsize=15)
                    ax4.set_xticks(range(0, 24, 3))
                    ax4.tick_params(axis='x', labelsize=15)
                    ax4.tick_params(axis='y', labelsize=15)

                    # Подсветка ночных часов
                    for hour in range(0, 6):
                        ax4.axvspan(hour, hour + 1, alpha=0.1, color='gray')
                else:
                    ax4.text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)

            # Настраиваем отступы между графиками
            plt.tight_layout()

            # Сохраняем с высоким разрешением
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return chart_data

        except Exception as e:
            print(f"Ошибка создания графиков: {e}")
            # Создаем заглушку с сообщением об ошибке
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Ошибка при создании графиков\n{str(e)[:100]}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_facecolor('#f8f9fa')
            ax.set_xticks([])
            ax.set_yticks([])

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return chart_data

generator = TransactionGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    stats = generator.stats.copy()
    stats['fraud_rate'] = stats['fraud'] / stats['total'] if stats['total'] > 0 else 0
    return jsonify(stats)

@app.route('/api/transactions/recent')
def get_recent_transactions():
    recent = generator.transactions_history[-10:] if generator.transactions_history else []
    for tx in recent:
        tx['amount'] = float(tx['amount'])
        tx['is_fraud'] = bool(tx['is_fraud'])
        if 'ml_confidence' in tx:
            tx['ml_confidence'] = float(tx['ml_confidence'])
    return jsonify(recent)

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    data = request.json
    speed = data.get('speed', 2.0)
    if not generator.is_running:
        generator.start_generation(speed)
        return jsonify({'status': 'started', 'speed': speed})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    generator.stop_generation()
    return jsonify({'status': 'stopped'})

@app.route('/api/train', methods=['POST'])
def train_model():
    success, message = generator.train_ml_model()
    return jsonify({'success': success, 'message': message})

@app.route('/api/charts')
def get_charts():
    chart_data = generator.get_chart_data()
    if chart_data:
        return jsonify({'chart': chart_data})
    return jsonify({'error': 'Нет данных для графиков'}), 400

@app.route('/api/export', methods=['POST'])
def export_data():
    try:
        if not generator.transactions_history:
            return jsonify({'error': 'Нет данных для экспорта'}), 400
        df = pd.DataFrame(generator.transactions_history)
        csv_data = df.to_csv(index=False, encoding='utf-8')
        return jsonify({'csv': csv_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_data():
    generator.transactions_history = []
    generator.stats = {'total': 0, 'normal': 0, 'fraud': 0, 'rule_blocked': 0, 'ml_blocked': 0}
    generator.ml_trained = False
    generator.transaction_count = 0
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("=" * 70)
    print("Запуск системы фрод-мониторинга")
    print("=" * 70)
    print("Научно-исследовательская работа на тему 'Фрод-мониторинг'")
    print("Выполнил: Серебренников Тимур | Группа: ПРИб-231")
    print("=" * 70)
    print("Откройте в браузере: http://localhost:5000")
    print("=" * 70)
    print("Инструкция:")
    print("1. Откройте сайт в браузере")
    print("2. Нажмите 'Запустить мониторинг'")
    print("3. Подождите 30 секунд для накопления данных")
    print("4. Нажмите 'Обучить ML-модель'")
    print("5. Нажмите 'Обновить графики'")
    print("=" * 70)
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)