# Music Recommendation MLOps

Система рекомендации музыки с полным MLOps циклом:
мониторинг дрейфа данных, автоматическое переобучение,
логирование экспериментов и деплой через GitOps.

---

### Data

Last.fm Dataset 1K users

---

## Архитектура


```
┌─────────────────────────────────────────────────────┐
│                    Kubernetes (Minikube)            │
│                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │   API   │  │   UI    │  │ MLflow  │              │
│  │ FastAPI │  │Streamlit│  │ Server  │              │
│  └────┬────┘  └────┬────┘  └────┬────┘              │
│       │            │            │                   │
│  ┌────▼────────────▼─────┐  ┌───▼─────┐             │
│  │      PostgreSQL       │  │   PVC   │             │
│  │  (users, drift, runs) │  │artifacts│             │
│  └───────────────────────┘  └─────────┘             │
│                                                     │
│  ┌──────────┐  ┌─────────┐  ┌─────────┐             │
│  │Prometheus│  │ Grafana │  │ Argo CD │             │
│  └──────────┘  └─────────┘  └─────────┘             │
└─────────────────────────────────────────────────────┘
```

### Компоненты

| Сервис | Назначение | Порт |
|--------|-----------|------|
| **API** (FastAPI) | Рекомендации, мониторинг дрейфа, переобучение | 8000 |
| **UI** (Streamlit) | Веб-интерфейс для пользователя и оператора | 8501 |
| **MLflow** | Логирование экспериментов и артефактов | 5000 |
| **PostgreSQL** | Хранение данных: пользователи, drift reports, runs | 5432 |
| **Prometheus** | Сбор метрик | 9090 |
| **Grafana** | Визуализация метрик | 3000 |
| **Argo CD** | GitOps деплой | 8080 |

---

## Модель

**Алгоритм:** ALS (Alternating Least Squares) через библиотеку `implicit`

**Данные:** Прослушивания треков (январь–март 2007)

**Переобучение:** Rolling window 45 дней при обнаружении дрейфа

**Метрики качества:**

| Метрика | Описание |
|---------|---------|
| NDCG@K | Normalized Discounted Cumulative Gain |
| Hit Rate@K | Доля пользователей с хотя бы одним попаданием |
| MRR@K | Mean Reciprocal Rank |
| Precision@K | Точность в топ-K |
| Recall@K | Полнота в топ-K |
| MAP@K | Mean Average Precision |

---

## Мониторинг

### Типы дрейфа

#### Data Drift
Сравнение распределений user-level фич текущего дня
с reference датасетом (январь + февраль)
через Evidently `DataDriftPreset`.

Отслеживаемые фичи:
- `n_interactions` - количество прослушиваний
- `n_unique_songs` - уникальных треков
- `avg_play_count` - среднее число прослушиваний
- `median_play_count` - медиана
- `std_play_count` - стандартное отклонение
- `max_play_count` - максимум
- `total_play_count` - сумма
- `diversity_ratio` - отношение уникальных треков к общему числу

#### Concept Drift
Деградация качества модели относительно baseline (`metrics.json`).
Порог: падение метрики > 10%.

Отслеживаемые метрики: NDCG@10, Hit Rate@10, MRR@10, Precision@10, Recall@10, MAP@10.

#### Target Drift
Сдвиг распределения популярности треков.
Метод: двухвыборочный тест Колмогорова-Смирнова
между распределениями популярности треков.

#### Statistical Anomaly
Выход значений за доверительный интервал (95%, t-распределение).
Требует минимум 7 дней истории.
Метрики: `n_interactions`, `n_unique_users`, `avg_play_count`.

### Prometheus метрики

| Метрика | Описание |
|---------|---------|
| `data_n_interactions` | Число взаимодействий за день |
| `data_n_users` | Число уникальных пользователей |
| `drift_score{drift_type}` | Score дрейфа по типу |
| `drift_detected{drift_type}` | Флаг обнаружения дрейфа (0/1) |
| `model_metric{metric_name}` | Метрики качества модели |
| `minutes_since_retrain` | Минут с последнего переобучения |

---


```markdown
## Быстрый старт

### Требования

- Docker Desktop
- Minikube
- kubectl
- Python 3.12
- Аккаунты: GitHub, DagsHub

---

### 1. Fork репозитория

> Обязательно делать Fork — CI/CD pipeline будет использовать твои собственные секреты и токены, не автора.

1. Открыть https://github.com/DianaBarmina/music-recommendation-mlops
2. Нажать **Fork** (правый верхний угол)
3. Создать fork в своём аккаунте

Клонировать свой fork:

```bash
git clone https://github.com/ВАШ_USERNAME/music-recommendation-mlops.git
cd music-recommendation-mlops
```

---

### 2. Настроить DVC remote (пример - я использовала DagsHub)

#### 2.1 Создать аккаунт и подключить репозиторий

1. Зарегистрироваться на [dagshub.com](https://dagshub.com) (через GitHub)
2. New Repository → Connect a repository → выбрать свой fork

#### 2.2 Получить DagsHub токен

1. DagsHub → аватар → User Settings → Developer Settings → Access Tokens
2. Скопировать **Default Access Token**

#### 2.3 Настроить DVC remote локально

```bash
pip install dvc

# Добавить DagsHub как remote
dvc remote add dagshub https://dagshub.com/ВАШ_USERNAME/music-recommendation-mlops.dvc
dvc remote default dagshub

# Добавить credentials (сохраняются локально, не в git)
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user ВАШ_DAGSHUB_USERNAME
dvc remote modify --local dagshub password ВАШ_DAGSHUB_TOKEN
```

---

### 3. Подготовить данные

#### 3.1 Структура данных

Проект ожидает данные в следующей структуре:

```
data/
├── raw/
│   ├── january/
│   │   └── interactions_january.parquet
│   ├── february/
│   │   └── interactions_february.parquet
│   └── march/
│       ├── march_01.parquet
│       ├── march_02.parquet
│       └── ... (до march_31.parquet)
```

Каждый файл содержит колонки:

| Колонка | Тип | Описание |
|---------|-----|---------|
| `user_id` | string | ID пользователя |
| `song_id` | string | ID трека |
| `ts` | datetime | Время прослушивания |
| `play_count` | int | Число прослушиваний (опционально) |

#### 3.2 Запустить DVC пайплайн

```bash
# Запустить полный пайплайн: очистка → фичи → обучение → оценка
dvc repro
```

Пайплайн создаст:
- `data/interim/clean_interactions.parquet`
- `data/processed/train_matrix.npz`
- `data/processed/users_map.parquet`
- `data/processed/items_map.parquet`
- `data/processed/reference/reference_dataset.parquet`
- `models/als_model.pkl`
- `metrics.json`

#### 3.3 Загрузить данные и модель в DagsHub

```bash
# Загрузить модель и необходимые артефакты
dvc push models/als_model.pkl
dvc push data/processed/users_map.parquet
dvc push data/processed/items_map.parquet
dvc push data/processed/train_matrix.npz
dvc push data/processed/reference/reference_dataset.parquet

# Загрузить сырые данные
dvc push data/raw/march
dvc push data/raw/january
dvc push data/raw/february
```

#### 3.4 Закоммитить изменения DVC конфига

```bash
git add .dvc/config
git commit -m "feat: add dagshub dvc remote"
git push origin main
```

---

### 4. Настроить GitHub секреты

В своём fork на GitHub → Settings → Secrets and variables → Actions → New repository secret:

| Секрет | Значение |
|--------|---------|
| `DOCKER_USERNAME` | Твой Docker Hub username |
| `DOCKER_PASSWORD` | Твой Docker Hub password или Access Token |

> Docker Hub: зарегистрироваться на [hub.docker.com](https://hub.docker.com)

---

### 5. Обновить манифесты под свой username

В файлах `k8s/app/api.yaml` и `k8s/app/ui.yaml` заменить username:

```yaml
# api.yaml
image: ВАШ_DOCKERHUB_USERNAME/music-recommendation-api:latest

# ui.yaml
image: ВАШ_DOCKERHUB_USERNAME/music-recommendation-ui:latest

# mlflow.yaml
image: ВАШ_DOCKERHUB_USERNAME/music-recommendation-mlflow:latest
```

В файле `k8s/argocd/application.yaml` заменить URL репозитория:

```yaml
repoURL: https://github.com/ВАШ_USERNAME/music-recommendation-mlops.git
```

```bash
git add k8s/
git commit -m "feat: update docker username and repo url"
git push origin main
```

Дождаться пока CI/CD pipeline соберёт и запушит образы на Docker Hub
(GitHub → Actions → наблюдать за прогрессом).

---

### 6. Создать файл авторизации для UI

Создать файл `services/ui/users.yaml`:

```yaml
users:
  - username: admin
    # Хэш генерируется командой:
    # python -c "from passlib.hash import bcrypt; print(bcrypt.hash('ВАШ_ПАРОЛЬ'))"
    hashed_password: "СГЕНЕРИРОВАННЫЙ_ХЭШ"
    role: admin
    full_name: "Administrator"
```

> Не коммитить `services/ui/users.yaml` в git — файл добавлен в `.gitignore`.
> Он будет передан в Kubernetes как Secret на шаге 9.

---

### 7. Запустить кластер

```bash
minikube start --driver=docker --cpus=4 --memory=6144 --disk-size=20g
minikube addons enable ingress
minikube addons enable metrics-server
```

---

### 8. Установить Argo CD

```bash
kubectl create namespace argocd

kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml \
  --server-side

# Дождаться всех подов (3-7 минут)
kubectl get pods -n argocd -w
```

---

### 9. Создать Kubernetes секреты

```bash
kubectl create namespace music-rec

# DagsHub credentials для скачивания модели и данных при старте
kubectl create secret generic dvc-credentials \
  --from-literal=DAGSHUB_USERNAME=ВАШ_DAGSHUB_USERNAME \
  --from-literal=DAGSHUB_TOKEN=ВАШ_DAGSHUB_TOKEN \
  -n music-rec

# Файл пользователей для авторизации в UI
kubectl create secret generic ui-users \
  --from-file=users.yaml=services/ui/users.yaml \
  -n music-rec
```

---

### 10. Подключить репозиторий в Argo CD

Открыть port-forward:

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

#### Получить пароль Argo CD

```bash
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}"

# Раскодировать (вставить своё значение вместо XXXXX)
python -c "import base64; print(base64.b64decode('XXXXX').decode())"
```

#### Создать GitHub Personal Access Token

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → выбрать scope `repo` → Generate
3. Скопировать токен — показывается только один раз

#### Подключить в Argo CD UI

Перейти на `https://localhost:8080` (принять предупреждение о сертификате):

```
Логин:  admin
Пароль: (из команды выше)
```

Settings → Repositories → Connect Repo:

```
Connection method: HTTPS
Repository URL: https://github.com/ВАШ_USERNAME/music-recommendation-mlops.git
Username: ВАШ_GITHUB_USERNAME
Password: ВАШ_GITHUB_TOKEN
```

---

### 11. Задеплоить приложение

```bash
kubectl apply -f k8s/argocd/application.yaml
```

Дождаться пока все поды поднимутся (5-10 минут):

```bash
kubectl get pods -n music-rec -w
```

Ожидаемый результат:

```
NAME               READY   STATUS    RESTARTS
api-XXXXX          1/1     Running   0
grafana-XXXXX      1/1     Running   0
mlflow-XXXXX       1/1     Running   0
postgres-XXXXX     1/1     Running   0
prometheus-XXXXX   1/1     Running   0
ui-XXXXX           1/1     Running   0
```

---

### 12. Открыть сервисы

Открыть отдельный терминал для каждого сервиса:

```bash
# Терминал 1 — Argo CD (уже открыт)
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Терминал 2 — API
kubectl port-forward svc/api-service -n music-rec 8000:8000

# Терминал 3 — UI
kubectl port-forward svc/ui-service -n music-rec 8501:8501

# Терминал 4 — MLflow
kubectl port-forward svc/mlflow-service -n music-rec 5000:5000

# Терминал 5 — Grafana
kubectl port-forward svc/grafana-service -n music-rec 3000:3000
```

| Сервис | URL | Логин |
|--------|-----|-------|
| Argo CD | https://localhost:8080 | admin / (из секрета) |
| API docs | http://localhost:8000/docs | — |
| UI | http://localhost:8501 | (из users.yaml) |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / admin_password |

---

## GitOps flow

```
Код → PR → merge в main
        ↓
GitHub Actions: lint → test → build → push образов в Docker Hub
        ↓
Argo CD: обнаруживает новый коммит → применяет манифесты
        ↓
Kubernetes: перезапускает поды с новым образом
```
