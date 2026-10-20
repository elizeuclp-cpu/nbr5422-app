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
# BANCO DE DADOS DOS CABOS (Peso em kg/m)
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

# ===================================================
# CONSTANTES DA NORMA
# ===================================================
RHO_PADRAO = 1.225  # kg/m³ (densidade padrão a 15°C e nível do mar)
P0 = 1013.0         # hPa
T0 = 20.0           # °C
Z_FL = 0.03         # Coeficiente de variação para frente lenta (Seção 9.4.1.1)

# ===================================================
# FUNÇÕES AUXILIARES
# ===================================================

def pressao_atmosferica(altitude):
    """Estimativa da pressão atmosférica em função da altitude (fórmula barométrica)"""
    return P0 * np.exp(-0.00012 * altitude)

def densidade_ar(temperatura, altitude):
    """
    Seção 4.5 - Densidade relativa e massa específica do ar.
    """
    p = pressao_atmosferica(altitude)
    delta = (p / P0) * ((273 + T0) / (273 + temperatura))
    rho = delta * RHO_PADRAO
    return delta, rho

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
    Us = tensao_kV
    numerador = k_cs * (np.sqrt(2) / np.sqrt(3)) * Us * F_sf
    denominador = 1080 * (k_aff / k_zfl) * k_g
    termo = numerador / denominador
    d_ff = 2.174 * np.exp(termo) - 1
    return max(d_ff, 0.0)

def calcular_angulo_balanco(q0, d_cabo, Lm, p_cond, Lp, K):
    if p_cond <= 0 or Lp <= 0:
        return 0.0
    argumento = K * q0 * d_cabo * Lm / (p_cond * Lp)
    return np.degrees(np.arctan(argumento))

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

    st.subheader("🌡️ Atmosfera (Seção 4.5)")
    altitude = st.number_input("Altitude local (m)", min_value=0.0, max_value=3000.0, value=0.0, step=50.0)
    temp_ar = st.number_input("Temperatura do ar (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=1.0)
    
    # Cálculo da densidade aqui para exibir na sidebar
    delta_ar, rho_ar = densidade_ar(temp_ar, altitude)
    st.caption(f"Pressão atm. estimada: {pressao_atmosferica(altitude):.1f} hPa")
    st.caption(f"Densidade relativa (\(\delta\)): {delta_ar:.4f}")
    st.caption(f"Massa específica (\(\mu\)): **{rho_ar:.4f} kg/m³**")

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
    k_cs = 1.35 # Fixado pela norma para cálculo da faixa (Seção 17.2.2.5)
    k_aff = st.number_input("K_aff (atmosférico)", min_value=0.8, max_value=1.2, value=1.0, step=0.01)
    k_zfl = 1 - 3 * Z_FL
    k_g = 1.28 # Fixado pela norma para cálculo da faixa (Seção 17.2.2.5)

    st.caption("ℹ️ Para cálculo da faixa de passagem, a Seção 17.2.2.5 fixa **K_CS = 1,35** e **k_g = 1,28**.")

    st.subheader("🏠 Obstáculos (Tabela 7)")
    tipo_obstaculo = st.selectbox("Se houver obstáculo na borda, qual o tipo?",
        ['Nenhum (Apenas elétrico)', 'Edificações', 'Veículos rodoviários e ferroviários', 'Vegetação de preservação permanente'])
    
    P_bH_map = {
        'Nenhum (Apenas elétrico)': 0.00,
        'Edificações': 4.00,
        'Veículos rodoviários e ferroviários': 1.10,
        'Vegetação de preservação permanente': 3.00
    }
    P_bH = P_bH_map[tipo_obstaculo]

# ===================================================
# CÁLCULOS GLOBAIS
# ===================================================
K_tur = fator_turbulencia(regiao)
K_rug = fator_rugosidade(categoria_terreno)

V_p10min = K_tur * V_R
V_p30s = fator_integracao_10min_para_30s() * K_rug * V_p10min
q0_30s = 0.5 * rho_ar * (V_p30s ** 2)  # Usa o rho calculado dinamicamente

P_e = parcela_eletrica(tensao_nominal, k_cs, k_aff, k_zfl, k_g, F_sf)
D_H_obstaculo = P_bH + P_e

# ===================================================
# ABAS
# ===================================================
aba1, aba2, aba3 = st.tabs(["📋 Dados Gerais e Elétricos", "📏 Faixa de Passagem", "📖 Memorial Resumido"])

# ===================================================
# ABA 1 - DADOS GERAIS E ELÉTRICOS
# ===================================================
with aba1:
    st.header("📋 Dados Gerais e Elétricos")
    st.markdown("Nesta aba, a **Parcela Elétrica (\(P_e\))** é calculada conforme a Seção 9.4.1. Ela é a base para a distância de segurança.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔌 Tensão e Fatores")
        st.metric("Tensão nominal", f"{tensao_nominal:.1f} kV")
        st.metric("Fator F_sf", f"{F_sf:.2f} p.u.")
        st.metric("K_CS (fixo p/ faixa)", f"{k_cs:.3f}")
        st.metric("K_aff", f"{k_aff:.3f}")
        st.metric("K_zfl (frente lenta)", f"{k_zfl:.3f}")
        st.metric("k_g (fixo p/ faixa)", f"{k_g:.3f}")

    with col2:
        st.subheader("🌬️ Vento e Atmosfera")
        st.metric("Altitude", f"{altitude:.0f} m")
        st.metric("Temperatura do ar", f"{temp_ar:.1f} °C")
        st.metric("Densidade do ar (\(\mu\))", f"{rho_ar:.4f} kg/m³")
        st.metric("V_p (10 min)", f"{V_p10min:.2f} m/s")
        st.metric("V_p (30 s)", f"{V_p30s:.2f} m/s")
        st.metric("q0 (30 s)", f"{q0_30s:.2f} Pa")

    st.markdown("---")
    st.subheader("🎯 Resultados das Distâncias")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("P_e (Parcela Elétrica)", f"{P_e:.3f} m")
        st.caption("Usada como 'D' na Faixa de Passagem (Seção 17)")
    with col_r2:
        st.metric("P_bH (Parcela Básica)", f"{P_bH:.2f} m")
        st.caption("Tabela 7. Zero se não houver obstáculo.")
    with col_r3:
        st.metric("D_H Obstáculo", f"{D_H_obstaculo:.3f} m")
        st.caption("D_H = P_bH + P_e (Seção 7)")

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
    - $D$ = distância horizontal de segurança (para a faixa, $D = P_e$)
    """)

    st.markdown("---")

    col_geo, col_res = st.columns([1, 1])

    with col_geo:
        st.subheader("🏗️ Geometria da Linha")

        b = st.number_input("b — distância horizontal do eixo ao condutor mais afastado (m)",
                            min_value=0.0, max_value=30.0, value=5.0, step=0.1)
        
        comprimento_cadeia = st.number_input("Comprimento da cadeia de isoladores (m)",
                                              min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        
        flecha = st.number_input("Flecha do condutor (m)",
                                 min_value=0.1, max_value=50.0, value=10.0, step=0.1)
                                 
        Lm = st.number_input("Vão médio L_m (m)", min_value=50.0, max_value=1500.0, value=300.0, step=10.0)
        Lp = st.number_input("Vão gravante L_p (m)", min_value=50.0, max_value=1500.0, value=300.0, step=10.0)

        st.subheader("🔌 Cabo Condutor")
        cabo_nome = st.selectbox("Tipo de cabo", list(CABOS_CONDUTORES.keys()))
        cabo = CABOS_CONDUTORES[cabo_nome]
        
        p_cond = cabo['peso'] * 9.81  
        d_cabo = cabo['D']
        
        st.caption(f"Peso linear: {p_cond:.4f} N/m | Diâmetro: {d_cabo*1000:.2f} mm")

        st.subheader("🌬️ Ângulo de Balanço (Seção 8.9)")
        K_bal = fator_efetividade_vento(V_p30s)
        beta_medio = calcular_angulo_balanco(q0_30s, d_cabo, Lm, p_cond, Lp, K_bal)
        sigma_beta = desvio_padrao_angulo(V_p30s)
        
        beta_max_calculado = beta_medio + 2 * sigma_beta
        beta_max = min(beta_max_calculado, 90.0)

        st.metric("K (Figura 22)", f"{K_bal:.3f}")
        st.metric("β médio", f"{beta_medio:.2f}°")
        st.metric("σ_β", f"{sigma_beta:.2f}°")
        st.metric("β max = β + 2σ", f"{beta_max:.2f}°")
        
        if beta_max_calculado > 90.0:
            st.warning(f"⚠️ O ângulo calculado ultrapassou 90° ({beta_max_calculado:.1f}°). O resultado foi limitado a 90°.")
        else:
            st.caption("ℹ️ O ângulo de balanço é calculado com a velocidade de rajada (30 s).")

    with col_res:
        st.subheader("📐 Projeções e Resultado")

        proj_flecha = flecha * np.sin(np.radians(beta_max))
        proj_cadeia = comprimento_cadeia * np.sin(np.radians(beta_max))
        d = proj_flecha + proj_cadeia

        st.metric("Projeção da flecha", f"{proj_flecha:.3f} m")
        st.metric("Projeção da cadeia", f"{proj_cadeia:.3f} m")
        st.metric("d = proj_flecha + proj_cadeia", f"{d:.3f} m")

        st.markdown("---")
        
        if tipo_obstaculo == 'Nenhum (Apenas elétrico)':
            D_usado = P_e
            st.metric("D (Parcela Elétrica)", f"{D_usado:.3f} m")
            st.caption("Usando apenas a distância elétrica (Seção 17).")
        else:
            D_usado = D_H_obstaculo
            st.metric("D (Com Obstáculo)", f"{D_usado:.3f} m")
            st.caption(f"Incluindo P_bH = {P_bH:.2f} m para {tipo_obstaculo}.")
            
        st.metric("b", f"{b:.2f} m")
        st.metric("d", f"{d:.3f} m")

        L = 2 * (b + d + D_usado)

        st.markdown("### 🎯 Largura Total da Faixa de Passagem")
        st.metric("L = 2(b + d + D)", f"{L:.2f} m", delta=f"≈ {L/2:.1f} m de cada lado do eixo")

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
            ax.text(-(b+d+L/2)/2, 11, f'D={D_usado:.1f}', ha='center', fontsize=9, color='purple')

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
    
    with st.expander("1. Velocidade de vento de 10 min para projeto (Seção 8.2.1)", expanded=False):
        bloco_memorial(
            titulo="1.1 Velocidade de 10 min para projeto",
            formula_latex=r"V_p = K_{tur} \cdot V_R",
            origem_coeficientes=f"""
            - $V_R$ = **{V_R:.1f} m/s** → informado pelo usuário
            - $K_{{tur}}$ = **{K_tur:.3f}** → **Tabela 8** (fator de turbulência da região *{regiao}*), valor fixado em norma
            """,
            substituicao_latex=rf"V_p = {K_tur:.3f} \cdot {V_R:.1f} = {V_p10min:.2f}\ \text{{m/s}}",
            resultado_latex=rf"\boxed{{V_p = {V_p10min:.2f}\ \text{{m/s}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.2.1.1 e Tabela 8"
        )

    with st.expander("2. Velocidade de vento de 30 s para projeto (Seção 8.2.2)", expanded=False):
        bloco_memorial(
            titulo="2.1 Velocidade de 30 s para projeto",
            formula_latex=r"V_{p30s} = K_{int} \cdot K_{alt} \cdot K_{rug} \cdot V_p",
            origem_coeficientes=f"""
            - $K_{{int}}$ = **{fator_integracao_10min_para_30s():.2f}** → **Tabela 9** (conversão de 10 min → 30 s)
            - $K_{{alt}}$ = **{fator_altura(10.0):.3f}** → calculado por $K_{{alt}} = (h/10)^{{0.14}}$, com $h = 10$ m
            - $K_{{rug}}$ = **{K_rug:.3f}** → **Tabela 1** (categoria do terreno *{categoria_terreno}*)
            - $V_p$ = **{V_p10min:.2f} m/s** → calculado no item 1
            """,
            substituicao_latex=rf"V_{{p30s}} = {fator_integracao_10min_para_30s():.2f} \cdot {fator_altura(10.0):.3f} \cdot {K_rug:.3f} \cdot {V_p10min:.2f} = {V_p30s:.2f}\ \text{{m/s}}",
            resultado_latex=rf"\boxed{{V_{{p30s}} = {V_p30s:.2f}\ \text{{m/s}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.2.2.1, Tabelas 1 e 9"
        )

    # ITEM 3 ATUALIZADO COM O CÁLCULO DA DENSIDADE DO AR
    with st.expander("3. Densidade do ar e Pressão dinâmica do vento (Seções 4.5 e 8.4.2)", expanded=True):
        st.markdown("#### 3.1 Densidade relativa do ar $\\delta$")
        st.caption("📘 ABNT NBR 5422:2024 — Seção 4.5.1")
        st.latex(r"\delta = \frac{p}{p_0} \left( \frac{273 + t_0}{273 + t} \right)")
        st.markdown(f"""
        - $p$ = **{pressao_atmosferica(altitude):.1f} hPa** → pressão local estimada para altitude de {altitude:.0f} m
        - $p_0$ = **1013 hPa** → pressão de referência
        - $t$ = **{temp_ar:.1f} °C** → temperatura do ar informada
        - $t_0$ = **20 °C** → temperatura de referência
        """)
        st.latex(rf"\delta = \frac{{{pressao_atmosferica(altitude):.1f}}}{{1013}} \left( \frac{{273 + 20}}{{273 + {temp_ar:.1f}}} \right) = {delta_ar:.4f}")
        
        st.markdown("#### 3.2 Massa específica do ar $\\mu$")
        st.latex(r"\mu = \delta \cdot \mu_0")
        st.markdown(f"Onde $\\mu_0$ = 1,225 kg/m³ (valor padrão a 15°C e nível do mar).")
        st.latex(rf"\mu = {delta_ar:.4f} \cdot 1.225 = {rho_ar:.4f}\ \text{{kg/m}}^3")
        st.latex(rf"\boxed{{\mu = {rho_ar:.4f}\ \text{{kg/m}}^3}}")

        st.markdown("---")
        st.markdown("#### 3.3 Pressão dinâmica $q_0$")
        st.caption("📘 ABNT NBR 5422:2024 — Seção 8.4.2")
        st.latex(r"q_0 = \frac{1}{2} \mu V_{p30s}^2")
        st.markdown(f"""
        - $\\mu$ = **{rho_ar:.4f} kg/m³** → calculado no item 3.2
        - $V_{{p30s}}$ = **{V_p30s:.2f} m/s** → calculado no item 2
        """)
        st.latex(rf"q_0 = \frac{{1}}{{2}} \cdot {rho_ar:.4f} \cdot ({V_p30s:.2f})^2 = {q0_30s:.2f}\ \text{{Pa}}")
        st.latex(rf"\boxed{{q_0 = {q0_30s:.2f}\ \text{{Pa}}}}")
        st.markdown("---")

    with st.expander("4. Parcela elétrica da distância de segurança (Seção 9.4.1.1)", expanded=False):
        bloco_memorial(
            titulo="4.1 Parcela elétrica $P_e$",
            formula_latex=r"P_e = 2{,}174 \cdot \exp\left[\frac{K_{cs} \frac{\sqrt{2}}{\sqrt{3}} U_s F_{sf}}{1080 \frac{K_{aff}}{K_{zfl}} k_g}\right] - 1",
            origem_coeficientes=f"""
            - $U_s$ = **{tensao_nominal:.1f} kV** → informado
            - $F_{{sf}}$ = **{F_sf:.2f}** → informado (Tabela 14)
            - $K_{{cs}}$ = **{k_cs:.3f}** → **Fixado pela norma** para Faixa de Passagem (Seção 17.2.2.5)
            - $K_{{aff}}$ = **{k_aff:.3f}** → informado (fator atmosférico)
            - $K_{{zfl}}$ = **{k_zfl:.3f}** → calculado p
