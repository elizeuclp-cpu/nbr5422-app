import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===================================================
# CONFIGURAÇÃO DA PÁGINA
# ===================================================
st.set_page_config(
    page_title="Cálculo de Faixa de Passagem - NBR 5422:2024",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Cálculo de Faixa de Passagem — NBR 5422:2024")
st.caption("Cálculo da largura da faixa de passagem conforme a Seção 17.2.2 da ABNT NBR 5422:2024")

# ===================================================
# BANCO DE DADOS DOS CABOS
# ===================================================
CABOS_CONDUTORES = {
    'Grosbeak': {'peso': 1.3028, 'area': 374.8, 'E': 7593, 'alpha': 189e-7, 'CR': 11427, 'D': 0.02515},
    'Linnet':   {'peso': 0.6883, 'area': 198.0, 'E': 7593, 'alpha': 189e-7, 'CR': 6393,  'D': 0.0183},
    'Penguin':  {'peso': 0.4330, 'area': 125.06, 'E': 8120, 'alpha': 186e-7, 'CR': 3790,  'D': 0.01431},
    'Cairo AAAC 6201': {'peso': 0.651, 'area': 236.38, 'E': 8120, 'alpha': 186e-7, 'CR': 7106.3, 'D': 0.0199},
    'Raven':    {'peso': 0.2170, 'area': 125.09, 'E': 7593, 'alpha': 189e-7, 'CR': 1985,  'D': 0.01011},
    'Flint':    {'peso': 1.0350, 'area': 374.52, 'E': 8120, 'alpha': 186e-7, 'CR': 11012.8, 'D': 0.0251},
    'Drake':    {'peso': 1.6290, 'area': 468.0, 'E': 7593, 'alpha': 189e-7, 'CR': 14245, 'D': 0.02813},
}

CABOS_PARA_RAIOS = {
    'EHS 5/16"': {'CR': 4900, 'peso': 0.305, 'area': 38.32, 'E': 18500, 'alpha': 11.5e-6, 'D': 0.00794},
    'EHS 3/8"':  {'CR': 6990, 'peso': 0.407, 'area': 51.14, 'E': 18500, 'alpha': 11.5e-6, 'D': 0.00952},
}

# ===================================================
# CONSTANTES DA NORMA
# ===================================================
RHO_AR = 1.225  # kg/m³
MU_AR = 1.45e-5  # m²/s
Z_FL = 0.03     # Coeficiente de variação para frente lenta (Seção 9.4.1.1)

# ===================================================
# FUNÇÕES AUXILIARES
# ===================================================

def fator_turbulencia(regiao):
    return {'Sul': 1.08, 'Sudeste e Centro-Oeste': 1.12, 'Norte e Nordeste': 1.16}[regiao]

def fator_rugosidade(categoria):
    return {'A': 1.08, 'B': 1.00, 'C': 0.85, 'D': 0.67}[categoria]

def fator_altura(h, z_ref=10.0):
    return (h / z_ref) ** 0.14

def fator_integracao_10min_para_30s():
    return 1.30

def fator_efetividade_vento(Vp):
    if Vp < 10:
        return 1.0
    return 0.3 + 3.68 * np.exp(-0.163 * Vp)

def desvio_padrao_angulo(Vp):
    return 2.25 * (1 - np.exp(-(Vp**2) / 230))

def parcela_eletrica(tensao_kV, k_cs, k_aff, k_zfl, k_g, F_sf):
    """
    Seção 9.4.1.1 - Parcela elétrica da distância de segurança (frente lenta).
    Fórmula exata da NBR 5422:2024.
    """
    Us = tensao_kV
    # Termo dentro da exponencial conforme a norma
    numerador = k_cs * (np.sqrt(2) / np.sqrt(3)) * Us * F_sf
    denominador = 1080 * (k_aff / k_zfl) * k_g
    
    termo = numerador / denominador
    d_ff = 2.174 * np.exp(termo) - 1
    
    return max(d_ff, 0.0)

def calcular_distancia_horizontal(tensao_kV, k_cs, k_aff, k_zfl, k_g, P_bH, F_sf):
    """
    Seção 7.3.2 - Distância horizontal de segurança
    """
    Pe = parcela_eletrica(tensao_kV, k_cs, k_aff, k_zfl, k_g, F_sf)
    return P_bH + Pe, Pe

def calcular_angulo_balanco(q0, d_cabo, Lm, p_cond, Lp, K):
    if p_cond <= 0 or Lp <= 0:
        return 0.0
    argumento = K * q0 * d_cabo * Lm / (p_cond * Lp)
    return np.degrees(np.arctan(argumento))

def pressao_dinamica(V):
    return 0.5 * RHO_AR * V**2

def bloco_memorial(titulo, formula_latex, origem_coeficientes, substituicao_latex, resultado_latex, referencia_norma):
    st.markdown(f"#### {titulo}")
    st.caption(f"📘 {referencia_norma}")
    st.markdown("**Fórmula:**")
    st.latex(formula_latex)
    st.markdown("**Origem dos coeficientes adotados:**")
    st.markdown(origem_coeficientes)
    st.markdown("**Substituição numérica:**")
    st.latex(substituicao_latex)
    st.markdown("**Resultado:**")
    st.latex(resultado_latex)
    st.markdown("---")

# ===================================================
# SIDEBAR - PARÂMETROS GLOBAIS
# ===================================================
with st.sidebar:
    st.header("🌍 Parâmetros Globais")

    tensao_nominal = st.number_input("Tensão nominal (kV)", min_value=38.0, max_value=800.0, value=138.0, step=1.0)
    fator_tensao_max = st.number_input("Fator de tensão máxima Ftmo (p.u.)", min_value=1.0, max_value=1.2, value=1.05, step=0.01)
    
    st.subheader("⚡ Sobretensão de Frente Lenta")
    F_sf = st.number_input("Fator de sobretensão F_sf (p.u.)", min_value=1.0, max_value=5.0, value=2.5, step=0.1, 
                           help="Tabela 14 da NBR 5422. Se não houver estudo específico, adotar valores típicos (ex: 2,5 para 138 kV).")

    st.subheader("🌬️ Vento")
    regiao = st.selectbox("Região geopolítica (Tabela 8)", ['Sudeste e Centro-Oeste', 'Sul', 'Norte e Nordeste'])
    categoria_terreno = st.selectbox("Categoria do terreno (Tabela 1)", ['B', 'A', 'C', 'D'],
                                     format_func=lambda x: {
                                         'A': 'A - Vastas extensões de água / áreas planas',
                                         'B': 'B - Terreno aberto com poucos obstáculos',
                                         'C': 'C - Terreno com obstáculos numerosos e pequenos',
                                         'D': 'D - Áreas urbanizadas / muitas árvores altas'
                                     }[x])
    V_R = st.number_input("Velocidade de referência V_R (m/s)", min_value=10.0, max_value=60.0, value=30.0, step=1.0)

    st.subheader("📐 Fatores de Segurança")
    k_cs = st.number_input("K_CS (Tabela 6)", min_value=1.0, max_value=2.0, value=1.35, step=0.01)
    k_aff = st.number_input("K_aff (atmosférico)", min_value=0.8, max_value=1.2, value=1.0, step=0.01)
    k_zff = st.number_input("K_zff", min_value=0.8, max_value=1.0, value=0.91, step=0.01)
    k_zfl = 1 - 3 * Z_FL  # 0.91

    st.subheader("🏠 Obstáculo (Tabela 7)")
    tipo_obstaculo = st.selectbox("Tipo de obstáculo",
        ['Edificações', 'Veículos rodoviários e ferroviários', 'Vegetação de preservação permanente'])
    P_bH_map = {
        'Edificações': (4.00, 1.28),
        'Veículos rodoviários e ferroviários': (1.10, 1.28),
        'Vegetação de preservação permanente': (3.00, 1.28)
    }
    P_bH, k_g = P_bH_map[tipo_obstaculo]

# ===================================================
# CÁLCULOS GLOBAIS
# ===================================================
K_tur = fator_turbulencia(regiao)
K_rug = fator_rugosidade(categoria_terreno)

V_p10min = K_tur * V_R
V_p30s = fator_integracao_10min_para_30s() * K_rug * V_p10min
q0_30s = pressao_dinamica(V_p30s)

# Cálculo da distância horizontal usando a tensão nominal
D_H, P_e = calcular_distancia_horizontal(tensao_nominal, k_cs, k_aff, k_zfl, k_g, P_bH, F_sf)

# ===================================================
# ABAS
# ===================================================
aba1, aba2, aba3 = st.tabs(["📋 Dados Gerais e Elétricos", "📏 Faixa de Passagem", "📖 Memorial Resumido"])

# ===================================================
# ABA 1 - DADOS GERAIS E ELÉTRICOS
# ===================================================
with aba1:
    st.header("📋 Dados Gerais e Elétricos")
    st.markdown("Nesta aba, os parâmetros elétricos são definidos e a **distância horizontal de segurança \(D_H\)** é calculada automaticamente conforme a Seção 7.3.2 da NBR 5422.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔌 Tensão e Fatores")
        st.metric("Tensão nominal", f"{tensao_nominal:.1f} kV")
        st.metric("Fator F_sf", f"{F_sf:.2f} p.u.")
        st.metric("Fator F_tmo", f"{fator_tensao_max:.3f}")
        st.metric("K_CS", f"{k_cs:.3f}")
        st.metric("K_aff", f"{k_aff:.3f}")
        st.metric("K_zfl (frente lenta)", f"{k_zfl:.3f}")
        st.metric("k_g (Tabela 7)", f"{k_g:.3f}")

    with col2:
        st.subheader("🌬️ Vento")
        st.metric("K_tur (Tabela 8)", f"{K_tur:.3f}")
        st.metric("K_rug (Tabela 1)", f"{K_rug:.3f}")
        st.metric("V_R", f"{V_R:.1f} m/s")
        st.metric("V_p (10 min)", f"{V_p10min:.2f} m/s")
        st.metric("V_p (30 s)", f"{V_p30s:.2f} m/s")
        st.metric("q0 (30 s)", f"{q0_30s:.2f} Pa")

    st.markdown("---")
    st.subheader("🎯 Resultado — Distância Horizontal de Segurança")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("P_bH (Tabela 7)", f"{P_bH:.2f} m")
    with col_r2:
        st.metric("P_e (parcela elétrica)", f"{P_e:.3f} m")
    with col_r3:
        st.metric("D_H = P_bH + P_e", f"{D_H:.3f} m", delta=f"Tipo: {tipo_obstaculo}")

    st.info(f"""
    **Fórmula aplicada (Seção 7.3.2):**

    $$D_H = P_{{bH}} + P_{{e}}$$

    - $P_{{bH}}$ = {P_bH:.2f} m (parcela básica, Tabela 7)
    - $P_{{e}}$ = {P_e:.3f} m (parcela elétrica, calculada em 9.4.1.1)

    **Portanto: $D_H$ = {D_H:.3f} m**
    """)

# ===================================================
# ABA 2 - FAIXA DE PASSAGEM
# ===================================================
with aba2:
    st.header("📏 Cálculo da Faixa de Passagem")
    st.markdown("Conforme a **Seção 17.2.2.5.1** da NBR 5422, a largura da faixa é calculada por:")

    st.latex(r"L = 2(b + d + D)")

    st.markdown("""
    Onde:
    - $b$ = distância horizontal do eixo do suporte ao ponto de fixação do condutor mais afastado
    - $d$ = soma das projeções horizontais da flecha do condutor + comprimento da cadeia de isoladores, considerando o ângulo de balanço $\\beta$
    - $D$ = distância horizontal de segurança (calculada na Aba 1)
    """)

    st.markdown("---")

    col_geo, col_res = st.columns([1, 1])

    with col_geo:
        st.subheader("🏗️ Geometria da Linha")

        b = st.number_input("b — distância horizontal do eixo ao condutor mais afastado (m)",
                            min_value=0.5, max_value=30.0, value=5.0, step=0.1)
        comprimento_cadeia = st.number_input("Comprimento da cadeia de isoladores (m)",
                                              min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        flecha = st.number_input("Flecha do condutor (m)",
                                 min_value=0.1, max_value=50.0, value=10.0, step=0.1)
        Lm = st.number_input("Vão médio L_m (m)", min_value=50.0, max_value=1000.0, value=300.0, step=10.0)
        Lp = st.number_input("Vão gravante L_p (m)", min_value=50.0, max_value=1000.0, value=300.0, step=10.0)

        st.subheader("🔌 Cabo Condutor")
        cabo_nome = st.selectbox("Tipo de cabo", list(CABOS_CONDUTORES.keys()))
        cabo = CABOS_CONDUTORES[cabo_nome]
        p_cond = cabo['peso'] * 9.81 / 1000  # N/m
        d_cabo = cabo['D']
        st.caption(f"Peso linear: {p_cond:.4f} N/m | Diâmetro: {d_cabo*1000:.2f} mm")

        st.subheader("🌬️ Ângulo de Balanço (Seção 8.9)")
        K_bal = fator_efetividade_vento(V_p30s)
        beta_medio = calcular_angulo_balanco(q0_30s, d_cabo, Lm, p_cond, Lp, K_bal)
        sigma_beta = desvio_padrao_angulo(V_p30s)
        beta_max = beta_medio + 2 * sigma_beta

        st.metric("K (Figura 22)", f"{K_bal:.3f}")
        st.metric("β médio", f"{beta_medio:.2f}°")
        st.metric("σ_β", f"{sigma_beta:.2f}°")
        st.metric("β max = β + 2σ", f"{beta_max:.2f}°")

    with col_res:
        st.subheader("📐 Projeções e Resultado")

        proj_flecha = flecha * np.sin(np.radians(beta_max))
        proj_cadeia = comprimento_cadeia * np.sin(np.radians(beta_max))
        d = proj_flecha + proj_cadeia

        st.metric("Projeção da flecha", f"{proj_flecha:.3f} m")
        st.metric("Projeção da cadeia", f"{proj_cadeia:.3f} m")
        st.metric("d = proj_flecha + proj_cadeia", f"{d:.3f} m")

        st.markdown("---")
        st.metric("D (da Aba 1)", f"{D_H:.3f} m")
        st.metric("b", f"{b:.2f} m")
        st.metric("d", f"{d:.3f} m")

        L = 2 * (b + d + D_H)

        st.markdown("### 🎯 Largura Total da Faixa de Passagem")
        st.metric("L = 2(b + d + D)", f"{L:.2f} m", delta=f"≈ {L/2:.1f} m de cada lado do eixo")

        st.info(f"""
        **Fórmula aplicada (Seção 17.2.2.5.1):**

        $$L = 2 \\cdot (b + d + D)$$

        - $b$ = {b:.2f} m
        - $d$ = {d:.3f} m (projeção da flecha + cadeia)
        - $D$ = {D_H:.3f} m

        **$L$ = {L:.2f} m**
        """)

    # Diagrama da faixa
    st.markdown("---")
    st.subheader("📈 Diagrama da Faixa de Passagem")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.axvspan(-L/2, L/2, alpha=0.15, color='blue', label=f'Faixa de passagem (L={L:.1f} m)')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Eixo da linha')

    ax.plot([0, 0], [0, 30], color='gray', linewidth=3)
    ax.plot([-1, 1], [30, 30], color='gray', linewidth=2)

    for sinal in [-1, 1]:
        x_rep = sinal * b
        x_bal = sinal * (b + d)
        ax.plot(x_rep, 25, 'ro', markersize=10, label='Condutor (repouso)' if sinal == 1 else '')
        ax.plot(x_bal, 25, 'r^', markersize=10, label='Condutor (balançado)' if sinal == 1 else '')
        ax.annotate('', xy=(x_rep, 23), xytext=(x_bal, 23),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        if sinal == 1:
            ax.text((x_rep + x_bal)/2, 24, f'd={d:.1f} m', ha='center', fontsize=9, color='red')

    ax.annotate('', xy=(-L/2, 5), xytext=(L/2, 5),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(0, 3, f'L = {L:.1f} m', ha='center', fontsize=11, color='blue', fontweight='bold')

    for sinal in [-1, 1]:
        ax.annotate('', xy=(sinal*b, 15), xytext=(sinal*(b+d), 15),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax.annotate('', xy=(sinal*(b+d), 10), xytext=(sinal*(L/2), 10),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
        if sinal == -1:
            ax.text(-b/2, 16, f'b={b:.1f}', ha='center', fontsize=9, color='green')
            ax.text(-(b+d+L/2)/2, 11, f'D={D_H:.1f}', ha='center', fontsize=9, color='purple')

    ax.set_xlim(-L/2 - 5, L/2 + 5)
    ax.set_ylim(0, 32)
    ax.set_xlabel('Coordenada horizontal (m)')
    ax.set_ylabel('Altura (m)')
    ax.set_title(f'Faixa de Passagem — L = {L:.2f} m')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ===================================================
# ABA 3 - MEMORIAL RESUMIDO
# ===================================================
with aba3:
    st.header("📖 Memorial Resumido de Cálculo")
    st.markdown("""
    Este memorial apresenta, para cada grandeza calculada:
    1. A **fórmula** aplicada;
    2. A **origem dos coeficientes** (tabela/figura da norma ou cálculo);
    3. A **substituição numérica** passo a passo;
    4. O **resultado final**.

    Todos os coeficientes são rastreáveis à NBR 5422:2024.
    """)

    # 1. VELOCIDADE DE VENTO DE 10 MIN
    with st.expander("1. Velocidade de vento de 10 min para projeto (Seção 8.2.1)", expanded=True):
        bloco_memorial(
            titulo="1.1 Velocidade de 10 min para projeto",
            formula_latex=r"V_p = K_{tur} \cdot V_R",
            origem_coeficientes=f"""
            - $V_R$ = **{V_R:.1f} m/s** → informado pelo usuário (velocidade de referência, 4.9.3.3)
            - $K_{{tur}}$ = **{K_tur:.3f}** → **Tabela 8** (fator de turbulência da região *{regiao}*), valor fixado em norma, **não calculado**
            """,
            substituicao_latex=rf"V_p = {K_tur:.3f} \cdot {V_R:.1f} = {V_p10min:.2f}\ \text{{m/s}}",
            resultado_latex=rf"\boxed{{V_p = {V_p10min:.2f}\ \text{{m/s}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.2.1.1 e Tabela 8"
        )

    # 2. VELOCIDADE DE 30 S
    with st.expander("2. Velocidade de vento de 30 s para projeto (Seção 8.2.2)", expanded=True):
        bloco_memorial(
            titulo="2.1 Velocidade de 30 s para projeto",
            formula_latex=r"V_{p30s} = K_{int} \cdot K_{alt} \cdot K_{rug} \cdot V_p",
            origem_coeficientes=f"""
            - $K_{{int}}$ = **{fator_integracao_10min_para_30s():.2f}** → **Tabela 9** (conversão de 10 min → 30 s), valor fixado em norma, **não calculado**
            - $K_{{alt}}$ = **{fator_altura(10.0):.3f}** → calculado por $K_{{alt}} = (h/10)^{{0.14}}$, com $h = 10$ m (referência), portanto $= 1{{,}}000$
            - $K_{{rug}}$ = **{K_rug:.3f}** → **Tabela 1** (categoria do terreno *{categoria_terreno}*), valor fixado em norma, **não calculado**
            - $V_p$ = **{V_p10min:.2f} m/s** → calculado no item 1
            """,
            substituicao_latex=rf"V_{{p30s}} = {fator_integracao_10min_para_30s():.2f} \cdot {fator_altura(10.0):.3f} \cdot {K_rug:.3f} \cdot {V_p10min:.2f} = {V_p30s:.2f}\ \text{{m/s}}",
            resultado_latex=rf"\boxed{{V_{{p30s}} = {V_p30s:.2f}\ \text{{m/s}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.2.2.1, Tabelas 1 e 9"
        )

    # 3. PRESSÃO DINÂMICA
    with st.expander("3. Pressão dinâmica do vento (Seção 8.4.2)", expanded=True):
        bloco_memorial(
            titulo="3.1 Pressão dinâmica $q_0$",
            formula_latex=r"q_0 = \frac{1}{2} \rho V_{p30s}^2",
            origem_coeficientes=f"""
            - $\\rho$ = **{RHO_AR} kg/m³** → massa específica do ar (Seção 4.7), valor de referência adotado, **não calculado**
            - $V_{{p30s}}$ = **{V_p30s:.2f} m/s** → calculado no item 2
            """,
            substituicao_latex=rf"q_0 = \frac{{1}}{{2}} \cdot {RHO_AR} \cdot ({V_p30s:.2f})^2 = {q0_30s:.2f}\ \text{{Pa}}",
            resultado_latex=rf"\boxed{{q_0 = {q0_30s:.2f}\ \text{{Pa}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.4.2"
        )

    # 4. PARCELA ELÉTRICA (AGORA CORRIGIDA)
    with st.expander("4. Parcela elétrica da distância de segurança (Seção 9.4.1.1)", expanded=True):
        bloco_memorial(
            titulo="4.1 Parcela elétrica $P_e$",
            formula_latex=r"P_e = 2{,}174 \cdot \exp\left[\frac{K_{cs} \frac{\sqrt{2}}{\sqrt{3}} U_s F_{sf}}{1080 \frac{K_{aff}}{K_{zfl}} k_g}\right] - 1",
            origem_coeficientes=f"""
            - $U_s$ = **{tensao_nominal:.1f} kV** → informado (tensão nominal)
            - $F_{{sf}}$ = **{F_sf:.2f}** → informado (Tabela 14)
            - $K_{{cs}}$ = **{k_cs:.3f}** → informado (Tabela 6)
            - $K_{{aff}}$ = **{k_aff:.3f}** → informado (fator atmosférico, Seção 4.8)
            - $K_{{zfl}}$ = **{k_zfl:.3f}** → calculado por $K_{{zfl}} = 1 - 3 z_{{fl}}$, com $z_{{fl}} = 0{{,}}03$ → $1 - 3 \\cdot 0{{,}}03 = {k_zfl:.2f}$
            - $k_g$ = **{k_g:.3f}** → **Tabela 7** (fator de espaçamento para *{tipo_obstaculo}*), valor fixado em norma, **não calculado**
            - Constante **1080** → coeficiente empírico da norma (rigidez dielétrica do ar em kV/m), **não calculado**
            - Constante **2,174** → coeficiente empírico da norma (Seção 9.4.1.1), **não calculado**
            """,
            substituicao_latex=rf"P_e = 2{{,}}174 \cdot \exp\left[\frac{{{k_cs:.3f} \cdot \frac{{\sqrt{{2}}}}{{\sqrt{{3}}}} \cdot {tensao_nominal:.1f} \cdot {F_sf:.2f}}}{{1080 \cdot \frac{{{k_aff:.3f}}}{{{k_zfl:.2f}}} \cdot {k_g:.3f}}}\right] - 1 = {P_e:.3f}\ \text{{m}}",
            resultado_latex=rf"\boxed{{P_e = {P_e:.3f}\ \text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seções 9.4.1.1, Tabelas 6, 7 e 14"
        )

    # 5. DISTÂNCIA HORIZONTAL DE SEGURANÇA
    with st.expander("5. Distância horizontal de segurança (Seção 7.3.2)", expanded=True):
        bloco_memorial(
            titulo="5.1 Distância horizontal de segurança $D_H$",
            formula_latex=r"D_H = P_{bH} + P_e",
            origem_coeficientes=f"""
            - $P_{{bH}}$ = **{P_bH:.2f} m** → **Tabela 7** (parcela básica para *{tipo_obstaculo}*), valor fixado em norma, **não calculado**
            - $P_e$ = **{P_e:.3f} m** → calculado no item 4
            """,
            substituicao_latex=rf"D_H = {P_bH:.2f} + {P_e:.3f} = {D_H:.3f}\ \text{{m}}",
            resultado_latex=rf"\boxed{{D_H = {D_H:.3f}\ \text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 7.3.2, Tabela 7"
        )

    # 6. FATOR DE EFETIVIDADE DO VENTO
    with st.expander("6. Fator de efetividade do vento para ângulo de balanço (Figura 22)", expanded=True):
        bloco_memorial(
            titulo="6.1 Fator $K$",
            formula_latex=r"K = 0{,}3 + 3{,}68 \cdot \exp(-0{,}163 \cdot V_p)\quad \text{para } V_p \ge 10\ \text{m/s}",
            origem_coeficientes=f"""
            - $V_p$ = **{V_p30s:.2f} m/s** → calculado no item 2
            - Constantes **0,3 / 3,68 / 0,163** → coeficientes empíricos da **Figura 22**, valores fixados em norma, **não calculados**
            """,
            substituicao_latex=rf"K = 0{{,}}3 + 3{{,}}68 \cdot \exp(-0{{,}}163 \cdot {V_p30s:.2f}) = {K_bal:.3f}",
            resultado_latex=rf"\boxed{{K = {K_bal:.3f}}}",
            referencia_norma="ABNT NBR 5422:2024 — Figura 22 (Seção 8.9.1)"
        )

    # 7. ÂNGULO DE BALANÇO MÉDIO
    with st.expander("7. Ângulo de balanço médio (Seção 8.9.1)", expanded=True):
        bloco_memorial(
            titulo="7.1 Ângulo de balanço médio $\\beta$",
            formula_latex=r"\beta = \arctan\left(K \cdot \frac{q_0 \cdot d \cdot L_m}{p_{cond} \cdot L_p}\right)",
            origem_coeficientes=f"""
            - $K$ = **{K_bal:.3f}** → calculado no item 6
            - $q_0$ = **{q0_30s:.2f} Pa** → calculado no item 3
            - $d$ = **{d_cabo*1000:.2f} mm** = **{d_cabo:.5f} m** → diâmetro do cabo *{cabo_nome}* (banco de dados)
            - $L_m$ = **{Lm:.1f} m** → informado (vão médio)
            - $p_{{cond}}$ = **{p_cond:.4f} N/m** → peso linear do cabo *{cabo_nome}* = {cabo['peso']:.4f} kg/m × 9,81 / 1000
            - $L_p$ = **{Lp:.1f} m** → informado (vão gravante)
            """,
            substituicao_latex=rf"\beta = \arctan\left({K_bal:.3f} \cdot \frac{{{q0_30s:.2f} \cdot {d_cabo:.5f} \cdot {Lm:.1f}}}{{{p_cond:.4f} \cdot {Lp:.1f}}}\right) = {beta_medio:.2f}^\circ",
            resultado_latex=rf"\boxed{{\beta = {beta_medio:.2f}^\circ}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.9.1"
        )

    # 8. DESVIO-PADRÃO DO ÂNGULO
    with st.expander("8. Desvio-padrão do ângulo de balanço (Seção 8.9.3)", expanded=True):
        bloco_memorial(
            titulo="8.1 Desvio-padrão $\\sigma_\\beta$",
            formula_latex=r"\sigma_\beta = 2{,}25 \cdot \left[1 - \exp\left(-\frac{V_p^2}{230}\right)\right]",
            origem_coeficientes=f"""
            - $V_p$ = **{V_p30s:.2f} m/s** → calculado no item 2
            - Constantes **2,25** e **230** → coeficientes empíricos da **Seção 8.9.3**, valores fixados em norma, **não calculados**
            """,
            substituicao_latex=rf"\sigma_\beta = 2{{,}}25 \cdot \left[1 - \exp\left(-\frac{{({V_p30s:.2f})^2}}{{230}}\right)\right] = {sigma_beta:.2f}^\circ",
            resultado_latex=rf"\boxed{{\sigma_\beta = {sigma_beta:.2f}^\circ}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.9.3"
        )

    # 9. ÂNGULO MÁXIMO
    with st.expander("9. Ângulo máximo de balanço (Seção 8.9.4)", expanded=True):
        bloco_memorial(
            titulo="9.1 Ângulo máximo $\\beta_{max}$",
            formula_latex=r"\beta_{max} = \beta + 2\sigma_\beta",
            origem_coeficientes=f"""
            - $\\beta$ = **{beta_medio:.2f}°** → calculado no item 7
            - $\\sigma_\\beta$ = **{sigma_beta:.2f}°** → calculado no item 8
            - Fator **2** → dois desvios-padrão, fixado em norma (Seção 8.9.4), **não calculado**
            """,
            substituicao_latex=rf"\beta_{{max}} = {beta_medio:.2f} + 2 \cdot {sigma_beta:.2f} = {beta_max:.2f}^\circ",
            resultado_latex=rf"\boxed{{\beta_{{max}} = {beta_max:.2f}^\circ}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.9.4"
        )

    # 10. PROJEÇÕES E d
    with st.expander("10. Projeções horizontais e parcela $d$ (Seção 17.2.2)", expanded=True):
        bloco_memorial(
            titulo="10.1 Parcela $d$ (projeção da flecha + cadeia)",
            formula_latex=r"d = f \cdot \sin(\beta_{max}) + L_{cad} \cdot \sin(\beta_{max})",
            origem_coeficientes=f"""
            - $f$ = **{flecha:.2f} m** → informado (flecha do condutor)
            - $L_{{cad}}$ = **{comprimento_cadeia:.2f} m** → informado (comprimento da cadeia de isoladores)
            - $\\beta_{{max}}$ = **{beta_max:.2f}°** → calculado no item 9
            """,
            substituicao_latex=rf"d = {flecha:.2f} \cdot \sin({beta_max:.2f}^\circ) + {comprimento_cadeia:.2f} \cdot \sin({beta_max:.2f}^\circ) = {proj_flecha:.3f} + {proj_cadeia:.3f} = {d:.3f}\ \text{{m}}",
            resultado_latex=rf"\boxed{{d = {d:.3f}\ \text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 17.2.2.5.1"
        )

    # 11. LARGURA DA FAIXA
    with st.expander("11. Largura total da faixa de passagem (Seção 17.2.2.5.1)", expanded=True):
        bloco_memorial(
            titulo="11.1 Largura $L$",
            formula_latex=r"L = 2 \cdot (b + d + D)",
            origem_coeficientes=f"""
            - $b$ = **{b:.2f} m** → informado (distância do eixo ao condutor mais afastado)
            - $d$ = **{d:.3f} m** → calculado no item 10
            - $D$ = **{D_H:.3f} m** → calculado no item 5 (distância horizontal de segurança)
            - Fator **2** → simetria em relação ao eixo da linha, fixado em norma (Seção 17.2.2.5.1), **não calculado**
            """,
            substituicao_latex=rf"L = 2 \cdot ({b:.2f} + {d:.3f} + {D_H:.3f}) = 2 \cdot {b+d+D_H:.3f} = {L:.2f}\ \text{{m}}",
            resultado_latex=rf"\boxed{{L = {L:.2f}\ \text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 17.2.2.5.1"
        )

    # RESUMO FINAL
    st.markdown("---")
    st.subheader("📋 Resumo Final")
    df_resumo = pd.DataFrame({
        "Grandeza": ["V_p (10 min)", "V_p (30 s)", "q_0", "P_e", "D_H", "β", "σ_β", "β_max", "d", "L"],
        "Valor": [
            f"{V_p10min:.2f} m/s",
            f"{V_p30s:.2f} m/s",
            f"{q0_30s:.2f} Pa",
            f"{P_e:.3f} m",
            f"{D_H:.3f} m",
            f"{beta_medio:.2f}°",
            f"{sigma_beta:.2f}°",
            f"{beta_max:.2f}°",
            f"{d:.3f} m",
            f"{L:.2f} m"
        ],
        "Origem": [
            "Calculado (item 1)",
            "Calculado (item 2)",
            "Calculado (item 3)",
            "Calculado (item 4)",
            "Calculado (item 5)",
            "Calculado (item 7)",
            "Calculado (item 8)",
            "Calculado (item 9)",
            "Calculado (item 10)",
            "Calculado (item 11)"
        ]
    })
    st.dataframe(df_resumo, hide_index=True, use_container_width=True)

    st.caption("⚠️ Este app é uma ferramenta de apoio. Os resultados devem ser verificados por engenheiro responsável conforme a NBR 5422:2024.")
