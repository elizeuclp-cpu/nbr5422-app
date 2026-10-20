import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="ZFS - Faixa de Servidão",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("⚡ ZFS - Faixa de Passagem e Servidão")
st.caption(
    "ABNT NBR 5422:2024 - Módulo inicial para cálculo de faixa de passagem."
)

# ============================================================
# DICIONÁRIOS
# ============================================================

K_RUG = {
    "A": 1.08,
    "B": 1.00,
    "C": 0.85,
    "D": 0.67
}

K_TUR = {
    "Sul": 1.08,
    "Sudeste/Centro-Oeste": 1.12,
    "Norte/Nordeste": 1.16
}

K_INT_30S = {
    "A": 1.17,
    "B": 1.22,
    "C": 1.30,
    "D": 1.48
}

# ============================================================
# FUNÇÕES
# ============================================================

def fator_altura(h, categoria):
    alfa = {
        "A": 0.110,
        "B": 0.160,
        "C": 0.220,
        "D": 0.280
    }

    return (h / 10) ** alfa[categoria]


def velocidade_30s(vr, regiao, categoria, altura):
    k_tur = K_TUR[regiao]

    vp = vr * k_tur

    k_alt = fator_altura(altura, categoria)

    k_int = K_INT_30S[categoria]

    vp30 = vp * k_alt * k_int

    return vp, vp30, k_alt, k_int


def pressao_dinamica(mu, vp30):
    return 0.5 * mu * vp30 ** 2


def fator_efetividade(vp30):
    if vp30 < 10:
        return 1
    return 3.68 * np.exp(-0.163 / vp30)


def angulo_balanco(q0,
                   diametro,
                   vao_medio,
                   k,
                   peso_linear,
                   vao_gravante):

    tg_beta = (
        q0 *
        diametro *
        vao_medio *
        k
    ) / (
        peso_linear *
        vao_gravante
    )

    beta = np.degrees(np.arctan(tg_beta))

    return beta


def sigma_beta(vp30):
    return 2.25 * (
        1 - np.exp(-(vp30**2)/230)
    )


# ============================================================
# ABAS
# ============================================================

aba1, aba2 = st.tabs(
    [
        "📋 Dados Gerais",
        "📐 Faixa de Passagem"
    ]
)

# ============================================================
# ABA 1
# ============================================================

with aba1:

    st.subheader("📋 Identificação")

    col1, col2 = st.columns(2)

    with col1:
        empreendimento = st.text_input(
            "Empreendimento"
        )

        linha = st.text_input(
            "Linha"
        )

        trecho = st.text_input(
            "Trecho"
        )

    with col2:
        responsavel = st.text_input(
            "Responsável"
        )

    st.markdown("---")

    st.subheader("⚡ Dados Elétricos")

    col1, col2 = st.columns(2)

    with col1:

        tensao_nominal = st.selectbox(
            "Tensão nominal (kV)",
            [69, 138, 230, 345, 500]
        )

    with col2:

        ftmo = st.number_input(
            "Ftmo",
            value=1.05
        )

    st.markdown("---")

    st.subheader("🌬️ Dados Meteorológicos")

    col1, col2, col3 = st.columns(3)

    with col1:

        regiao = st.selectbox(
            "Região",
            [
                "Sul",
                "Sudeste/Centro-Oeste",
                "Norte/Nordeste"
            ]
        )

    with col2:

        categoria = st.selectbox(
            "Categoria do terreno",
            ["A", "B", "C", "D"]
        )

    with col3:

        vr = st.number_input(
            "Velocidade VR (m/s)",
            value=30.0
        )

    col1, col2 = st.columns(2)

    with col1:

        temperatura_max = st.number_input(
            "Temperatura máxima (°C)",
            value=40.0
        )

    with col2:

        mu = st.number_input(
            "Massa específica do ar (kg/m³)",
            value=1.225
        )

    st.markdown("---")

    st.subheader("📐 Geometria")

    col1, col2 = st.columns(2)

    with col1:

        diametro_cabo = st.number_input(
            "Diâmetro do cabo (m)",
            value=0.02515,
            format="%.5f"
        )

        peso_linear = st.number_input(
            "Peso linear (N/m)",
            value=15.20
        )

        altura_media = st.number_input(
            "Altura média do cabo (m)",
            value=25.0
        )

    with col2:

        vao_medio = st.number_input(
            "Vão médio Lm (m)",
            value=350.0
        )

        vao_gravante = st.number_input(
            "Vão gravante Lp (m)",
            value=330.0
        )

        flecha = st.number_input(
            "Flecha (m)",
            value=10.0
        )

        cadeia = st.number_input(
            "Comprimento da cadeia (m)",
            value=2.5
        )

# ============================================================
# ABA 2
# ============================================================

with aba2:

    st.subheader("📐 Faixa de Passagem")

    tipo_arranjo = st.radio(
        "Arranjo das fases",
        [
            "Simétrico",
            "Vertical"
        ]
    )

    if tipo_arranjo == "Simétrico":

        b = st.number_input(
            "Distância eixo-condutor externo b (m)",
            value=4.0
        )

    else:

        b = 0.0

    D = st.number_input(
        "Distância horizontal de segurança DH (m)",
        value=2.50
    )

    st.markdown("---")

    if st.button(
        "⚡ Calcular Faixa"
    ):

        vp, vp30, k_alt, k_int = velocidade_30s(
            vr,
            regiao,
            categoria,
            altura_media
        )

        q0 = pressao_dinamica(
            mu,
            vp30
        )

        k = fator_efetividade(
            vp30
        )

        beta = angulo_balanco(
            q0,
            diametro_cabo,
            vao_medio,
            k,
            peso_linear,
            vao_gravante
        )

        sigma = sigma_beta(vp30)

        beta_max = beta + 2 * sigma

        d_faixa = (
            flecha +
            cadeia
        ) * np.sin(
            np.radians(beta_max)
        )

        if tipo_arranjo == "Simétrico":

            largura = 2 * (
                b +
                d_faixa +
                D
            )

        else:

            largura = 2 * (
                d_faixa +
                D
            )

        # ---------------------------------------------------
        # RESULTADOS
        # ---------------------------------------------------

        st.markdown("---")
        st.subheader("📊 Resultados")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Vento 30s",
                f"{vp30:.2f} m/s"
            )

        with c2:

            st.metric(
                "βmáx",
                f"{beta_max:.2f}°"
            )

        with c3:

            st.metric(
                "Deslocamento horizontal",
                f"{d_faixa:.2f} m"
            )

        st.metric(
            "Faixa total",
            f"{largura:.2f} m"
        )

        st.metric(
            "Semi-faixa",
            f"{largura/2:.2f} m"
        )

        # ---------------------------------------------------
        # MEMORIAL
        # ---------------------------------------------------

        with st.expander(
            "📖 Memorial de cálculo"
        ):

            st.markdown("### 1 - Velocidade de vento")

            st.latex(
                r"V_{P30s}"
                r"="
                r"K_{int}"
                r"\cdot"
                r"K_{alt}"
                r"\cdot"
                r"V_P"
            )

            st.write(
                f"VP30s = {vp30:.2f} m/s"
            )

            st.markdown("### 2 - Pressão dinâmica")

            st.latex(
                r"q_0="
                r"\frac{1}{2}"
                r"\mu"
                r"V^2"
            )

            st.write(
                f"q0 = {q0:.2f} Pa"
            )

            st.markdown("### 3 - Ângulo de balanço")

            st.write(
                f"β = {beta:.2f}°"
            )

            st.write(
                f"σβ = {sigma:.2f}°"
            )

            st.write(
                f"βmáx = {beta_max:.2f}°"
            )

            st.markdown("### 4 - Deslocamento horizontal")

            st.latex(
                r"d=(f+l_c)"
                r"\sin(\beta_{max})"
            )

            st.write(
                f"d = {d_faixa:.2f} m"
            )

            st.markdown("### 5 - Largura da faixa")

            if tipo_arranjo == "Simétrico":

                st.latex(
                    r"L=2(b+d+D)"
                )

            else:

                st.latex(
                    r"L=2(d+D)"
                )

            st.write(
                f"L = {largura:.2f} m"
            )
