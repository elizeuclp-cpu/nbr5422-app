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
RHO_PADRAO = 1.225  # kg/m³
P0 = 1013.0         # hPa
T0 = 20.0           # °C
Z_FL = 0.03         # Coeficiente de variação para frente lenta

# ===================================================
# FUNÇÕES AUXILIARES
# ===================================================

def pressao_atmosferica(altitude):
    return P0 * np.exp(-0.00012 * altitude)

def densidade_ar(temperatura, altitude):
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
                           help="Tabela 14 da NBR 5422. Se não houver estudo específico, adotar valores típicos.")

    st.subheader("🌡️ Atmosfera (Seção 4.5)")
    altitude = st.number_input("Altitude local (m)", min_value=0.0, max_value=3000.0, value=0.0, step=50.0)
    temp_ar = st.number_input("Temperatura do ar (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=1.0)
    
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
    k_cs = 1.35
    k_aff = st.number_input("K_aff (atmosférico)", min_value=0.8, max_value=1.2, value=1.0, step=0.01)
    k_zfl = 1 - 3 * Z_FL
    k_g = 1.28

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
q0_30s = 0.5 * rho_ar * (V_p30s ** 2)

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
    
    # 1. VELOCIDADE DE VENTO DE 10 MIN
    with st.expander("1. Velocidade de vento de 10 min para projeto (Seção 8.2.1)", expanded=False):
        texto_origem = "- $V_R$ = **" + f"{V_R:.1f}" + " m/s** → informado pelo usuário\n- $K_{tur}$ = **" + f"{K_tur:.3f}" + "** → Tabela 8 (fator de turbulência da região *" + regiao + "*), valor fixado em norma"
        bloco_memorial(
            titulo="1.1 Velocidade de 10 min para projeto",
            formula_latex=r"V_p = K_{tur} \cdot V_R",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"V_p = {K_tur:.3f} \\cdot {V_R:.1f} = {V_p10min:.2f}\\ \\text{{m/s}}",
            resultado_latex=f"\\boxed{{V_p = {V_p10min:.2f}\\ \\text{{m/s}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.2.1.1 e Tabela 8"
        )

    # 2. VELOCIDADE DE 30 S
    with st.expander("2. Velocidade de vento de 30 s para projeto (Seção 8.2.2)", expanded=False):
        texto_origem = (
            "- $K_{int}$ = **" + f"{fator_integracao_10min_para_30s():.2f}" + "** → Tabela 9 (conversão de 10 min → 30 s)\n"
            "- $K_{alt}$ = **" + f"{fator_altura(10.0):.3f}" + "** → calculado por $K_{alt} = (h/10)^{0.14}$, com $h = 10$ m\n"
            "- $K_{rug}$ = **" + f"{K_rug:.3f}" + "** → Tabela 1 (categoria do terreno *" + categoria_terreno + "*)\n"
            "- $V_p$ = **" + f"{V_p10min:.2f}" + " m/s** → calculado no item 1"
        )
        bloco_memorial(
            titulo="2.1 Velocidade de 30 s para projeto",
            formula_latex=r"V_{p30s} = K_{int} \cdot K_{alt} \cdot K_{rug} \cdot V_p",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"V_{{p30s}} = {fator_integracao_10min_para_30s():.2f} \\cdot {fator_altura(10.0):.3f} \\cdot {K_rug:.3f} \\cdot {V_p10min:.2f} = {V_p30s:.2f}\\ \\text{{m/s}}",
            resultado_latex=f"\\boxed{{V_{{p30s}} = {V_p30s:.2f}\\ \\text{{m/s}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.2.2.1, Tabelas 1 e 9"
        )

    # 3. DENSIDADE DO AR E PRESSÃO DINÂMICA
    with st.expander("3. Densidade do ar e Pressão dinâmica do vento (Seções 4.5 e 8.4.2)", expanded=True):
        st.markdown("#### 3.1 Densidade relativa do ar $\\delta$")
        st.caption("📘 ABNT NBR 5422:2024 — Seção 4.5.1")
        st.latex(r"\delta = \frac{p}{p_0} \left( \frac{273 + t_0}{273 + t} \right)")
        
        st.markdown("- $p$ = **" + f"{pressao_atmosferica(altitude):.1f}" + " hPa** → pressão local estimada para altitude de " + f"{altitude:.0f}" + " m")
        st.markdown("- $p_0$ = **1013 hPa** → pressão de referência")
        st.markdown("- $t$ = **" + f"{temp_ar:.1f}" + " °C** → temperatura do ar informada")
        st.markdown("- $t_0$ = **20 °C** → temperatura de referência")
        
        st.latex(f"\\delta = \\frac{{{pressao_atmosferica(altitude):.1f}}}{{1013}} \\left( \\frac{{273 + 20}}{{273 + {temp_ar:.1f}}} \\right) = {delta_ar:.4f}")
        
        st.markdown("#### 3.2 Massa específica do ar $\\mu$")
        st.latex(r"\mu = \delta \cdot \mu_0")
        st.markdown("Onde $\\mu_0$ = 1,225 kg/m³ (valor padrão a 15°C e nível do mar).")
        st.latex(f"\\mu = {delta_ar:.4f} \\cdot 1.225 = {rho_ar:.4f}\\ \\text{{kg/m}}^3")
        st.latex(f"\\boxed{{\\mu = {rho_ar:.4f}\\ \\text{{kg/m}}^3}}")

        st.markdown("---")
        st.markdown("#### 3.3 Pressão dinâmica $q_0$")
        st.caption("📘 ABNT NBR 5422:2024 — Seção 8.4.2")
        st.latex(r"q_0 = \frac{1}{2} \mu V_{p30s}^2")
        st.markdown("- $\\mu$ = **" + f"{rho_ar:.4f}" + " kg/m³** → calculado no item 3.2")
        st.markdown("- $V_{p30s}$ = **" + f"{V_p30s:.2f}" + " m/s** → calculado no item 2")
        st.latex(f"q_0 = \\frac{{1}}{{2}} \\cdot {rho_ar:.4f} \\cdot ({V_p30s:.2f})^2 = {q0_30s:.2f}\\ \\text{{Pa}}")
        st.latex(f"\\boxed{{q_0 = {q0_30s:.2f}\\ \\text{{Pa}}}}")
        st.markdown("---")

    # 4. PARCELA ELÉTRICA
    with st.expander("4. Parcela elétrica da distância de segurança (Seção 9.4.1.1)", expanded=False):
        texto_origem = (
            "- $U_s$ = **" + f"{tensao_nominal:.1f}" + " kV** → informado\n"
            "- $F_{sf}$ = **" + f"{F_sf:.2f}" + "** → informado (Tabela 14)\n"
            "- $K_{cs}$ = **" + f"{k_cs:.3f}" + "** → Fixado pela norma para Faixa de Passagem (Seção 17.2.2.5)\n"
            "- $K_{aff}$ = **" + f"{k_aff:.3f}" + "** → informado (fator atmosférico)\n"
            "- $K_{zfl}$ = **" + f"{k_zfl:.3f}" + "** → calculado por $K_{zfl} = 1 - 3 z_{fl}$, com $z_{fl} = 0,03$\n"
            "- $k_g$ = **" + f"{k_g:.3f}" + "** → Fixado pela norma para Faixa de Passagem (Seção 17.2.2.5)\n"
            "- Constante **1080** → coeficiente empírico da norma\n"
            "- Constante **2,174** → coeficiente empírico da norma"
        )
        bloco_memorial(
            titulo="4.1 Parcela elétrica $P_e$",
            formula_latex=r"P_e = 2{,}174 \cdot \exp\left[\frac{K_{cs} \frac{\sqrt{2}}{\sqrt{3}} U_s F_{sf}}{1080 \frac{K_{aff}}{K_{zfl}} k_g}\right] - 1",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"P_e = 2{{,}}174 \\cdot \\exp\\left[\\frac{{{k_cs:.3f} \\cdot \\frac{{\\sqrt{{2}}}}{{\\sqrt{{3}}}} \\cdot {tensao_nominal:.1f} \\cdot {F_sf:.2f}}}{{1080 \\cdot \\frac{{{k_aff:.3f}}}{{{k_zfl:.2f}}} \\cdot {k_g:.3f}}}\\right] - 1 = {P_e:.3f}\\ \\text{{m}}",
            resultado_latex=f"\\boxed{{P_e = {P_e:.3f}\\ \\text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seções 9.4.1.1, Tabelas 6, 7 e 14"
        )

    # 5. DISTÂNCIA HORIZONTAL DE SEGURANÇA
    with st.expander("5. Distância horizontal de segurança (Seção 7.3.2)", expanded=False):
        texto_origem = (
            "- $P_{bH}$ = **" + f"{P_bH:.2f}" + " m** → Tabela 7 (parcela básica para *" + tipo_obstaculo + "*)\n"
            "- $P_e$ = **" + f"{P_e:.3f}" + " m** → calculado no item 4"
        )
        bloco_memorial(
            titulo="5.1 Distância horizontal de segurança $D_H$",
            formula_latex=r"D_H = P_{bH} + P_e",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"D_H = {P_bH:.2f} + {P_e:.3f} = {D_H_obstaculo:.3f}\\ \\text{{m}}",
            resultado_latex=f"\\boxed{{D_H = {D_H_obstaculo:.3f}\\ \\text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 7.3.2, Tabela 7"
        )

    # 6. FATOR DE EFETIVIDADE DO VENTO
    with st.expander("6. Fator de efetividade do vento para ângulo de balanço (Figura 22)", expanded=False):
        texto_origem = "- $V_p$ = **" + f"{V_p30s:.2f}" + " m/s** → calculado no item 2"
        bloco_memorial(
            titulo="6.1 Fator $K$",
            formula_latex=r"K = 0{,}3 + 3{,}68 \cdot \exp(-0{,}163 \cdot V_p)\quad \text{para } V_p \ge 10\ \text{m/s}",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"K = 0{{,}}3 + 3{{,}}68 \\cdot \\exp(-0{{,}}163 \\cdot {V_p30s:.2f}) = {K_bal:.3f}",
            resultado_latex=f"\\boxed{{K = {K_bal:.3f}}}",
            referencia_norma="ABNT NBR 5422:2024 — Figura 22 (Seção 8.9.1)"
        )

    # 7. ÂNGULO DE BALANÇO MÉDIO
    with st.expander("7. Ângulo de balanço médio (Seção 8.9.1)", expanded=False):
        texto_origem = (
            "- $K$ = **" + f"{K_bal:.3f}" + "** → calculado no item 6\n"
            "- $q_0$ = **" + f"{q0_30s:.2f}" + " Pa** → calculado no item 3.3\n"
            "- $d$ = **" + f"{d_cabo:.5f}" + " m** → diâmetro do cabo *" + cabo_nome + "*\n"
            "- $L_m$ = **" + f"{Lm:.1f}" + " m** → informado\n"
            "- $p_{cond}$ = **" + f"{p_cond:.4f}" + " N/m** → peso linear do cabo\n"
            "- $L_p$ = **" + f"{Lp:.1f}" + " m** → informado"
        )
        bloco_memorial(
            titulo="7.1 Ângulo de balanço médio $\\beta$",
            formula_latex=r"\beta = \arctan\left(K \cdot \frac{q_0 \cdot d \cdot L_m}{p_{cond} \cdot L_p}\right)",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"\\beta = \\arctan\\left({K_bal:.3f} \\cdot \\frac{{{q0_30s:.2f} \\cdot {d_cabo:.5f} \\cdot {Lm:.1f}}}{{{p_cond:.4f} \\cdot {Lp:.1f}}}\\right) = {beta_medio:.2f}^\\circ",
            resultado_latex=f"\\boxed{{\\beta = {beta_medio:.2f}^\\circ}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.9.1"
        )

    # 8. DESVIO-PADRÃO
    with st.expander("8. Desvio-padrão do ângulo de balanço (Seção 8.9.3)", expanded=False):
        texto_origem = "- $V_p$ = **" + f"{V_p30s:.2f}" + " m/s** → calculado no item 2"
        bloco_memorial(
            titulo="8.1 Desvio-padrão $\\sigma_\\beta$",
            formula_latex=r"\sigma_\beta = 2{,}25 \cdot \left[1 - \exp\left(-\frac{V_p^2}{230}\right)\right]",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"\\sigma_\\beta = 2{{,}}25 \\cdot \\left[1 - \\exp\\left(-\\frac{{({V_p30s:.2f})^2}}{{230}}\\right)\\right] = {sigma_beta:.2f}^\\circ",
            resultado_latex=f"\\boxed{{\\sigma_\\beta = {sigma_beta:.2f}^\\circ}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.9.3"
        )

    # 9. ÂNGULO MÁXIMO
    with st.expander("9. Ângulo máximo de balanço (Seção 8.9.4)", expanded=False):
        texto_origem = (
            "- $\\beta$ = **" + f"{beta_medio:.2f}" + "°** → calculado no item 7\n"
            "- $\\sigma_\\beta$ = **" + f"{sigma_beta:.2f}" + "°** → calculado no item 8"
        )
        bloco_memorial(
            titulo="9.1 Ângulo máximo $\\beta_{max}$",
            formula_latex=r"\beta_{max} = \min(\beta + 2\sigma_\beta, 90^\circ)",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"\\beta_{{max}} = \\min({beta_medio:.2f} + 2 \\cdot {sigma_beta:.2f}, 90^\\circ) = {beta_max:.2f}^\\circ",
            resultado_latex=f"\\boxed{{\\beta_{{max}} = {beta_max:.2f}^\\circ}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 8.9.4 (com limite físico de 90°)"
        )

    # 10. PROJEÇÕES E d
    with st.expander("10. Projeções horizontais e parcela $d$ (Seção 17.2.2)", expanded=False):
        texto_origem = (
            "- $f$ = **" + f"{flecha:.2f}" + " m** → informado\n"
            "- $L_{cad}$ = **" + f"{comprimento_cadeia:.2f}" + " m** → informado\n"
            "- $\\beta_{max}$ = **" + f"{beta_max:.2f}" + "°** → calculado no item 9"
        )
        bloco_memorial(
            titulo="10.1 Parcela $d$ (projeção da flecha + cadeia)",
            formula_latex=r"d = f \cdot \sin(\beta_{max}) + L_{cad} \cdot \sin(\beta_{max})",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"d = {flecha:.2f} \\cdot \\sin({beta_max:.2f}^\\circ) + {comprimento_cadeia:.2f} \\cdot \\sin({beta_max:.2f}^\\circ) = {d:.3f}\\ \\text{{m}}",
            resultado_latex=f"\\boxed{{d = {d:.3f}\\ \\text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 17.2.2.5.1"
        )

    # 11. LARGURA DA FAIXA
    with st.expander("11. Largura total da faixa de passagem (Seção 17.2.2.5.1)", expanded=False):
        texto_origem = (
            "- $b$ = **" + f"{b:.2f}" + " m** → informado\n"
            "- $d$ = **" + f"{d:.3f}" + " m** → calculado no item 10\n"
            "- $D$ = **" + f"{D_usado:.3f}" + " m** → calculado no item 5 (ou $P_e$)"
        )
        bloco_memorial(
            titulo="11.1 Largura $L$",
            formula_latex=r"L = 2 \cdot (b + d + D)",
            origem_coeficientes=texto_origem,
            substituicao_latex=f"L = 2 \\cdot ({b:.2f} + {d:.3f} + {D_usado:.3f}) = {L:.2f}\\ \\text{{m}}",
            resultado_latex=f"\\boxed{{L = {L:.2f}\\ \\text{{m}}}}",
            referencia_norma="ABNT NBR 5422:2024 — Seção 17.2.2.5.1"
        )

    st.markdown("---")
    st.subheader("📋 Resumo Final")
    df_resumo = pd.DataFrame({
        "Grandeza": ["V_p (10 min)", "V_p (30 s)", "Densidade do ar", "q_0", "P_e", "D_H (Obstáculo)", "β", "σ_β", "β_max", "d", "L"],
        "Valor": [
            f"{V_p10min:.2f} m/s",
            f"{V_p30s:.2f} m/s",
            f"{rho_ar:.4f} kg/m³",
            f"{q0_30s:.2f} Pa",
            f"{P_e:.3f} m",
            f"{D_H_obstaculo:.3f} m",
            f"{beta_medio:.2f}°",
            f"{sigma_beta:.2f}°",
            f"{beta_max:.2f}°",
            f"{d:.3f} m",
            f"{L:.2f} m"
        ],
        "Origem": [
            "Item 1", "Item 2", "Item 3.2", "Item 3.3", "Item 4",
            "Item 5", "Item 7", "Item 8", "Item 9", "Item 10", "Item 11"
        ]
    })
    st.dataframe(df_resumo, hide_index=True, use_container_width=True)

    st.caption("⚠️ Este app é uma ferramenta de apoio. Os resultados devem ser verificados por engenheiro responsável conforme a NBR 5422:2024.")
