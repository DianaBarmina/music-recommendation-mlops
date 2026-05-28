from pathlib import Path

import streamlit as st
import yaml
from passlib.hash import bcrypt

USERS_FILE = Path(__file__).parent / "users.yaml"

SESSION_TIMEOUT = 8 * 3600


def load_users() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    with open(USERS_FILE) as f:
        data = yaml.safe_load(f)
    return data.get("users", [])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.verify(plain_password, hashed_password)
    except Exception:
        return False


def authenticate(username: str, password: str) -> dict | None:
    users = load_users()
    for user in users:
        if user["username"] == username:
            if verify_password(password, user["hashed_password"]):
                return user
    return None


def login_page() -> None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🎵 Music Recommendation MLOps")
        st.markdown("### Вход в систему")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Логин",
                placeholder="Введите логин...",
                autocomplete="username",
            )
            password = st.text_input(
                "Пароль",
                type="password",
                placeholder="Введите пароль...",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button(
                "Войти",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if not username or not password:
                st.error("Введите логин и пароль")
                return

            user = authenticate(username, password)
            if user:
                import time

                st.session_state["authenticated"] = True
                st.session_state["username"] = user["username"]
                st.session_state["role"] = user["role"]
                st.session_state["full_name"] = user["full_name"]
                st.session_state["login_time"] = time.time()
                st.success(f"Добро пожаловать, {user['full_name']}!")
                st.rerun()
            else:
                st.error("Неверный логин или пароль")


def logout() -> None:
    keys_to_clear = [
        "authenticated",
        "username",
        "role",
        "full_name",
        "login_time",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.rerun()


def is_authenticated() -> bool:
    import time

    if not st.session_state.get("authenticated"):
        return False

    # Проверяем timeout сессии
    login_time = st.session_state.get("login_time", 0)
    if time.time() - login_time > SESSION_TIMEOUT:
        logout()
        return False

    return True


def require_auth() -> bool:
    """
    Главная функция — вызывается в начале main.py.
    Если не авторизован — показывает форму входа и возвращает False.
    Если авторизован — возвращает True и можно показывать приложение.
    """
    if not is_authenticated():
        login_page()
        return False
    return True


def show_user_info() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👤 **{st.session_state.get('full_name', 'Unknown')}**")
    st.sidebar.markdown(f"🔑 Роль: `{st.session_state.get('role', 'unknown')}`")
    if st.sidebar.button("🚪 Выйти", use_container_width=True):
        logout()
