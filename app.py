"""
=============================================================================
Landing Page — Cláudio Ferreira Neves
Apresentação do autor e entrada do curso de Machine Learning

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. | Jaraguá do Sul/SC
Docência    : Especialista de Ensino II — Análise de Dados | SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)

Como executar:
  streamlit run app.py
=============================================================================
"""

import streamlit as st

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Cláudio Ferreira Neves — Machine Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CSS
# ============================================================================

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Ocultar sidebar completamente */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero-section {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 60%, #1a0533 100%);
    border-radius: 20px;
    padding: 4rem 3rem 3.5rem;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
.hero-avatar {
    width: 140px; height: 140px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1.5rem;
    font-size: 2.5rem; font-weight: 700; color: white;
    border: 4px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(102,126,234,0.4);
}
.hero-name {
    font-size: 2.6rem; font-weight: 700; color: white;
    margin: 0 0 0.5rem; line-height: 1.2;
}
.hero-title {
    font-size: 1.15rem; color: rgba(255,255,255,0.8);
    margin: 0 0 0.6rem;
}
.hero-tagline {
    font-size: 1rem; font-style: italic;
    color: rgba(255,255,255,0.65); margin: 0 0 1.8rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 0.35rem 1rem;
    font-size: 0.88rem; margin: 0.3rem 0.25rem;
    color: rgba(255,255,255,0.9);
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px; padding: 1.4rem 1.5rem;
    color: white; text-align: center; margin-bottom: 0.5rem;
    box-shadow: 0 4px 20px rgba(102,126,234,0.35);
}
.metric-card h2 { margin: 0 0 0.3rem; font-size: 2.2rem; font-weight: 800; }
.metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }

/* ── Info / Tip / Success boxes ───────────────────────────────────────── */
.info-box {
    background: #f0f4ff; border-left: 5px solid #667eea;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
    font-size: 0.92rem; color: #1a1a2e; line-height: 1.6;
}
.tip-box {
    background: #fff8e1; border-left: 5px solid #f59e0b;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
    font-size: 0.92rem; color: #1a1a2e; line-height: 1.6;
}
.success-box {
    background: #f0fff4; border-left: 5px solid #38a169;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
    font-size: 0.92rem; color: #1a1a2e; line-height: 1.6;
}

/* ── Sobre o autor ────────────────────────────────────────────────────── */
.author-infobox {
    background: #f0f4ff; border-left: 5px solid #667eea;
    padding: 0.9rem 1.2rem; border-radius: 0 10px 10px 0;
    margin: 0.7rem 0; font-size: 0.9rem; color: #1a1a2e; line-height: 1.6;
}

/* ── Card escuro ──────────────────────────────────────────────────────── */
.dark-card {
    background: #1a1a2e; border-radius: 16px;
    padding: 2rem; color: white; height: 100%;
}
.dark-card h4 { margin: 0 0 1rem; font-size: 1.05rem; color: rgba(255,255,255,0.9); }
.dark-card ul { padding-left: 0; list-style: none; margin: 0 0 1.5rem; }
.dark-card ul li { padding: 0.35rem 0; font-size: 0.92rem; color: rgba(255,255,255,0.82); }

/* ── Experiência profissional ─────────────────────────────────────────── */
.exp-card {
    background: white; border-radius: 14px;
    border: 1px solid #e8ecf4; padding: 1.8rem 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07); height: 100%;
}
.exp-badge-indigo {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border-radius: 20px; padding: 0.3rem 1rem;
    font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem;
}
.exp-badge-green {
    display: inline-block;
    background: linear-gradient(135deg, #065f46, #059669);
    color: white; border-radius: 20px; padding: 0.3rem 1rem;
    font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem;
}
.exp-role {
    font-size: 1.05rem; font-weight: 700; color: #1a1a2e;
    margin: 0 0 0.5rem;
}
.exp-desc {
    font-size: 0.9rem; color: #555; line-height: 1.65; margin-bottom: 1rem;
}
.skill-tag {
    display: inline-block; background: #f0f4ff;
    border: 1px solid #c7d2fe; color: #3730a3;
    border-radius: 6px; padding: 0.15rem 0.65rem;
    font-size: 0.78rem; margin: 0.2rem 0.15rem 0 0;
}

/* ── Competências / pills ─────────────────────────────────────────────── */
.skill-pill {
    display: inline-block; background: #f0f4ff;
    border: 1px solid #c7d2fe; color: #3730a3;
    border-radius: 20px; padding: 0.4rem 1rem;
    font-size: 0.9rem; margin: 0.3rem 0.25rem;
}

/* ── Diferenciais ─────────────────────────────────────────────────────── */
.diff-card {
    background: white; border-radius: 14px;
    border: 1px solid #e8ecf4; padding: 1.8rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07); height: 100%;
    text-align: center;
}
.diff-card .diff-icon {
    font-size: 2.4rem; margin-bottom: 0.7rem;
}
.diff-card h4 { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; margin: 0 0 0.6rem; }
.diff-card p  { font-size: 0.9rem; color: #555; line-height: 1.6; margin: 0; }

/* ── CTA final ────────────────────────────────────────────────────────── */
.cta-section {
    background: #1a1a2e;
    border-radius: 20px; padding: 3.5rem 2rem;
    text-align: center; color: white; margin: 2rem 0;
}
.cta-section h2 { font-size: 2rem; font-weight: 800; margin: 0 0 0.6rem; }
.cta-section p  { font-size: 1.05rem; color: rgba(255,255,255,0.75); margin: 0 0 2rem; }

/* ── Footer ───────────────────────────────────────────────────────────── */
.footer {
    background: #1a1a2e; color: white;
    padding: 1.8rem 2rem; border-radius: 12px;
    text-align: center; margin-top: 3rem;
}
.footer p { margin: 0.25rem 0; font-size: 0.88rem; opacity: 0.75; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SEÇÃO 1 — HERO
# ============================================================================

st.markdown("""
<div class="hero-section">
    <div class="hero-avatar">CFN</div>
    <div class="hero-name">Cláudio Ferreira Neves</div>
    <div class="hero-title">Especialista em Ciência de Dados &amp; IA · Business Intelligence</div>
    <div class="hero-tagline">"Da Engenharia de Redes à Inteligência Artificial — uma jornada de dados em constante evolução"</div>
    <div>
        <span class="hero-badge">📅 10+ anos de experiência</span>
        <span class="hero-badge">🏢 SENAI/PA · SENAI/SC</span>
        <span class="hero-badge">🎓 Faculdade Pitágoras</span>
        <span class="hero-badge">🏛️ IFPA</span>
        <span class="hero-badge">🏫 SEDUC/PA</span>
    </div>
</div>
""", unsafe_allow_html=True)

_, col_cta, _ = st.columns([2, 2, 2])
with col_cta:
    if st.button("🚀 Acessar as Aulas →", type="primary", use_container_width=False, key="hero_cta"):
        st.switch_page("pages/Portal.py")

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# SEÇÃO 2 — MÉTRICAS
# ============================================================================

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><h2>10+</h2><p>Anos de Experiência</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><h2>100+</h2><p>Alunos Formados</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><h2>DACP · PL300</h2><p>Certificações Internacionais</p></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><h2>8</h2><p>Aulas no Curso</p></div>', unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# SEÇÃO 3 — SOBRE O AUTOR
# ============================================================================

st.markdown("## 👨‍💻 Sobre o Autor")

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("""
    <p style="font-size:0.97rem; color:#333; line-height:1.75; margin-bottom:1rem;">
    Cláudio Ferreira Neves começou sua trajetória na Engenharia de Redes de Telecomunicações e, ao
    longo de mais de 10 anos, construiu uma carreira sólida em dados — migrando do BI para Machine Learning
    e Inteligência Artificial. Acumula sete pós-graduações que refletem uma evolução constante e intencional:
    da infraestrutura de redes ao banco de dados, da gestão de projetos à ciência de dados.
    </p>
    <p style="font-size:0.97rem; color:#333; line-height:1.75; margin-bottom:1.5rem;">
    Especialista em Ciência de Dados e IA pela <b>uniSENAI</b> e em Business Intelligence, Big Data e Analytics
    pela <b>Faculdade Pitágoras</b>. Atuou como professor no SENAI do Estado do Pará em cursos técnicos e
    superiores de tecnologia, e hoje é professor de Análise de Dados no <b>SENAI/SC</b> e mentor do projeto
    <b>Lab365</b> do Estado de Santa Catarina — Análise de Dados e Inteligência Artificial.
    </p>

    <div class="author-infobox">
        🏆 <b>DACP &amp; PL300</b> — Data Analyst Certified Professional e Microsoft Power BI Data Analyst (PL-300),
        duas certificações internacionais em análise de dados e BI.
    </div>
    <div class="author-infobox">
        🏢 <b>SAVE Co. · PonceTech · Cinbesa · SENAI/PA · SENAI/SC</b> — Analista de BI na SAVE Co.,
        ex-Analista Pleno na PonceTech, ex-Gerente de Dados na Cinbesa/PA e professor no SENAI Marabá/PA
        e SENAI Jaraguá do Sul/SC. Mentor do Lab365/SC — Análise de Dados e IA.
    </div>
    <div class="author-infobox">
        🎓 <b>7 Pós-graduações</b> — De Engenharia de Redes a Ciência de Dados e IA, cada especialização
        representa uma etapa deliberada de crescimento e reposicionamento de carreira.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="dark-card">
        <h4>O que você vai encontrar neste curso:</h4>
        <ul>
            <li>✅ Machine Learning do zero ao avançado</li>
            <li>✅ 8 aulas com datasets reais</li>
            <li>✅ Código Python comentado linha a linha</li>
            <li>✅ Aplicações interativas com Streamlit</li>
            <li>✅ Metodologia CRISP-DM aplicada</li>
            <li>✅ Algoritmos supervisionados e não supervisionados</li>
            <li>✅ Projeto final com Voting Ensemble</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📚 Ver todas as aulas →", key="card_cta", use_container_width=True):
        st.switch_page("pages/Portal.py")

st.markdown("---")


# ============================================================================
# SEÇÃO 4 — EXPERIÊNCIA PROFISSIONAL
# ============================================================================

st.markdown("## 💼 Experiência Profissional")

# ── Mercado ──────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-size:0.8rem; font-weight:700; letter-spacing:1.2px; "
    "text-transform:uppercase; color:#667eea; margin-bottom:0.8rem;'>📊 Mercado</p>",
    unsafe_allow_html=True,
)

em1, em2, em3 = st.columns(3, gap="medium")

with em1:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-indigo">💼 SAVE Co. · Atual</div>
        <div class="exp-role">Analista de BI — SAVE Co.</div>
        <div class="exp-desc">
            Atualmente atua como Analista de Business Intelligence na SAVE Co., desenvolvendo
            dashboards estratégicos, pipelines de dados e soluções analíticas para apoio à
            tomada de decisão.
        </div>
        <div>
            <span class="skill-tag">Power BI</span>
            <span class="skill-tag">SQL</span>
            <span class="skill-tag">Python</span>
            <span class="skill-tag">ETL</span>
            <span class="skill-tag">DAX</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with em2:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-green">🔍 PonceTech Consultoria</div>
        <div class="exp-role">Analista de BI Pleno — PonceTech Consultoria</div>
        <div class="exp-desc">
            Atuou como Analista de BI Pleno na PonceTech Consultoria, entregando soluções de
            inteligência de negócios para clientes, com foco em modelagem de dados, relatórios
            analíticos e automação de processos de BI.
        </div>
        <div>
            <span class="skill-tag">Power BI</span>
            <span class="skill-tag">SQL Server</span>
            <span class="skill-tag">Modelagem de Dados</span>
            <span class="skill-tag">ETL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with em3:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-indigo">🏭 Cinbesa · Belém/PA</div>
        <div class="exp-role">Gerente de Dados — Cinbesa, Belém/PA</div>
        <div class="exp-desc">
            Gerenciou a área de dados da Cinbesa em Belém do Pará, liderando iniciativas de
            governança de dados, Data Warehouse e cultura data-driven na organização.
        </div>
        <div>
            <span class="skill-tag">Gestão de Dados</span>
            <span class="skill-tag">Data Warehouse</span>
            <span class="skill-tag">Governança</span>
            <span class="skill-tag">BI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

em4, _, _ = st.columns(3, gap="medium")

with em4:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-indigo">📊 Grupo Líder · Belém/PA · Projeto Temporário</div>
        <div class="exp-role">Analista de Dados Pleno — Controladoria · Grupo Líder</div>
        <div class="exp-desc">
            Projeto de 3 meses no maior grupo varejista do Pará (25+ anos de atuação).
            Implementou análise de dados no Protheus (TOTVS) — área que não tinha visibilidade
            analítica antes. Entregou dashboards financeiros, cruzamento de dados contábeis com
            SQL e Python, e automação do fechamento mensal na Controladoria.
        </div>
        <div>
            <span class="skill-tag">Power BI</span>
            <span class="skill-tag">SQL</span>
            <span class="skill-tag">Python</span>
            <span class="skill-tag">TOTVS Protheus</span>
            <span class="skill-tag">Controladoria</span>
            <span class="skill-tag">Auditoria de Dados</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)

# ── Docência ─────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-size:0.8rem; font-weight:700; letter-spacing:1.2px; "
    "text-transform:uppercase; color:#059669; margin-bottom:0.8rem;'>🎓 Docência</p>",
    unsafe_allow_html=True,
)

ed1, ed2, ed3 = st.columns(3, gap="medium")

with ed1:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-green">🎓 SENAI/SC</div>
        <div class="exp-role">Professor de Análise de Dados — SENAI Jaraguá do Sul/SC</div>
        <div class="exp-desc">
            Professor do curso de Análise de Dados no SENAI de Jaraguá do Sul, formando analistas
            com Power BI, SQL, Python e Machine Learning aplicados a projetos reais.
        </div>
        <div>
            <span class="skill-tag">Power BI</span>
            <span class="skill-tag">SQL</span>
            <span class="skill-tag">Python</span>
            <span class="skill-tag">Machine Learning</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ed2:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-green">🌐 Lab365 · SENAI/SC</div>
        <div class="exp-role">Mentor — Lab365 Estado de Santa Catarina</div>
        <div class="exp-desc">
            Mentor do projeto Lab365 do Estado de Santa Catarina pelo SENAI/SC, nas trilhas de
            Análise de Dados e Inteligência Artificial, orientando equipes em projetos práticos.
        </div>
        <div>
            <span class="skill-tag">Análise de Dados</span>
            <span class="skill-tag">Inteligência Artificial</span>
            <span class="skill-tag">Mentoria</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ed3:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-indigo">🏢 SENAI/PA</div>
        <div class="exp-role">Professor de Tecnologia — SENAI Marabá/PA</div>
        <div class="exp-desc">
            Ministrou cursos técnicos e superiores na área de tecnologia no SENAI do Pará,
            formando profissionais com foco em infraestrutura, redes e banco de dados.
        </div>
        <div>
            <span class="skill-tag">Redes</span>
            <span class="skill-tag">Banco de Dados</span>
            <span class="skill-tag">Ensino Técnico</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

ed4, ed5, _ = st.columns([1, 1, 1], gap="medium")

with ed4:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-green">🏫 SEDUC/PA</div>
        <div class="exp-role">Professor da Escola Técnica e Tecnologia — SEDUC/PA</div>
        <div class="exp-desc">
            Professor de tecnologia na rede estadual de ensino do Pará, ministrando disciplinas
            técnicas e contribuindo com a formação tecnológica pública no estado.
        </div>
        <div>
            <span class="skill-tag">Escola Técnica</span>
            <span class="skill-tag">Tecnologia</span>
            <span class="skill-tag">Ensino Público</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ed5:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-indigo">🏛️ IFPA</div>
        <div class="exp-role">Professor Substituto de TI — IFPA</div>
        <div class="exp-desc">
            Professor Substituto de Tecnologia da Informação no Instituto Federal do Pará,
            atuando no ensino técnico e superior federal da área de TI.
        </div>
        <div>
            <span class="skill-tag">TI</span>
            <span class="skill-tag">Ensino Federal</span>
            <span class="skill-tag">IFPA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

ed6, _, _ = st.columns([1, 1, 1], gap="medium")

with ed6:
    st.markdown("""
    <div class="exp-card">
        <div class="exp-badge-green">🎓 Faculdade Pitágoras</div>
        <div class="exp-role">Professor de Nível A — Ensino Superior · Faculdade Pitágoras</div>
        <div class="exp-desc">
            Professor de Nível A no Ensino Superior da Faculdade Pitágoras, ministrando
            disciplinas na área de tecnologia e ciência de dados.
        </div>
        <div>
            <span class="skill-tag">Ensino Superior</span>
            <span class="skill-tag">Tecnologia</span>
            <span class="skill-tag">Faculdade Pitágoras</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# SEÇÃO 4b — FORMAÇÃO ACADÊMICA
# ============================================================================

st.markdown("## 🎓 Formação Acadêmica")
st.markdown(
    "<p style='font-size:0.93rem; color:#666; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "Uma trajetória de especialização contínua — da infraestrutura de redes à inteligência artificial.</p>",
    unsafe_allow_html=True,
)

formacao = [
    ("#667eea", "🤖", "Ciência de Dados e Inteligência Artificial",
     "Pós-Graduação · uniSENAI  ✦ Atual", "Especialização em andamento integrando ciência de dados e IA — curso que originou este material."),
    ("#d97706", "📊", "Business Intelligence, Big Data e Analytics — Ciência de Dados",
     "Pós-Graduação · Universidade Pitágoras", "Consolidação em BI e big data, com foco em análise e ciência de dados aplicada ao negócio."),
    ("#dc2626", "🧠", "Ciência de Dados",
     "Pós-Graduação", "Aprofundamento em modelagem estatística, machine learning e pipelines de dados end-to-end."),
    ("#0f3460", "📋", "MBA em Gestão de Projetos",
     "MBA", "Visão estratégica e liderança de projetos — complementando o perfil técnico com competências de gestão."),
    ("#059669", "💻", "Tecnologias Digitais Aplicadas à Educação",
     "Pós-Graduação", "Fundamentação pedagógica para o ensino técnico e uso de tecnologia como ferramenta educacional."),
    ("#764ba2", "🗄️", "Administração em Banco de Dados",
     "Pós-Graduação", "Base em modelagem, SQL e gestão de bancos relacionais — fundamento técnico para a área de dados."),
    ("#667eea", "🔌", "Engenharia de Redes de Telecomunicações",
     "Pós-Graduação", "Ponto de partida da carreira técnica — infraestrutura e redes que antecederam a transição para dados."),
]

for cor, icon, titulo, tipo, desc in formacao:
    st.markdown(f"""
    <div style="display:flex; gap:1rem; align-items:flex-start; margin-bottom:0.65rem;
                background:white; border-radius:12px; border:1px solid #e8ecf4;
                padding:1rem 1.4rem; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <div style="width:42px; height:42px; border-radius:50%; flex-shrink:0;
                    background:{cor}; display:flex; align-items:center;
                    justify-content:center; font-size:1.2rem;">{icon}</div>
        <div style="flex:1;">
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.8px;
                        text-transform:uppercase; color:{cor}; margin-bottom:0.15rem;">{tipo}</div>
            <div style="font-size:0.97rem; font-weight:700; color:#1a1a2e;
                        margin-bottom:0.2rem;">{titulo}</div>
            <div style="font-size:0.85rem; color:#666; line-height:1.5;">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# SEÇÃO 5 — COMPETÊNCIAS TÉCNICAS
# ============================================================================

st.markdown("## ⚙️ Competências Técnicas")

skills = [
    "Power BI & DAX",
    "SQL & SQL Server",
    "Python (pandas, numpy, scikit-learn)",
    "Data Analytics & EDA",
    "ETL / Data Warehouse",
    "Machine Learning",
    "Inteligência Artificial",
    "LangChain",
    "Apache Spark",
    "Databricks",
    "AWS (Instrutor)",
    "Modelagem de Dados",
    "GitHub",
    "Streamlit",
    "Visualização de Dados",
    "CRISP-DM",
    "Automação de Processos",
    "Data Storytelling",
]

sc1, sc2, sc3 = st.columns(3)
cols_skills = [sc1, sc2, sc3]
for i, skill in enumerate(skills):
    with cols_skills[i % 3]:
        st.markdown(f'<span class="skill-pill">{skill}</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")


# ============================================================================
# SEÇÃO 6 — DIFERENCIAIS
# ============================================================================

st.markdown("## 🎯 Por que aprender com Cláudio?")

d1, d2, d3 = st.columns(3, gap="large")

with d1:
    st.markdown("""
    <div class="diff-card" style="border-top: 4px solid #667eea;">
        <div class="diff-icon">🔧</div>
        <h4>Aplicação Prática</h4>
        <p>Cada aula usa datasets reais e resolve problemas que você vai encontrar no mercado.
        Sem exemplos artificiais, sem teoria vazia.</p>
    </div>
    """, unsafe_allow_html=True)

with d2:
    st.markdown("""
    <div class="diff-card" style="border-top: 4px solid #a855f7;">
        <div class="diff-icon">🏭</div>
        <h4>Experiência de Mercado</h4>
        <p>10+ anos em projetos reais — de Analista de BI na SAVE Co. e PonceTech a Gerente
        de Dados na Cinbesa. O conteúdo reflete o que funciona na prática, não só em livros.</p>
    </div>
    """, unsafe_allow_html=True)

with d3:
    st.markdown("""
    <div class="diff-card" style="border-top: 4px solid #38a169;">
        <div class="diff-icon">🎯</div>
        <h4>Didática Comprovada</h4>
        <p>Metodologia testada com dezenas de alunos. Código comentado, conceitos explicados em
        linguagem clara, evolução gradual e consistente.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# SEÇÃO 7 — CTA FINAL
# ============================================================================

st.markdown("""
<div class="cta-section">
    <h2>Pronto para dominar Machine Learning?</h2>
    <p>8 aulas interativas, datasets reais e código Python comentado esperando por você.</p>
</div>
""", unsafe_allow_html=True)

_, col_final, _ = st.columns([1, 3, 1])
with col_final:
    if st.button(
        "🚀 Começar Agora — Acessar as Aulas",
        type="primary",
        use_container_width=True,
        key="final_cta",
    ):
        st.switch_page("pages/Portal.py")

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <p style="font-size:1rem; font-weight:600; opacity:0.9; margin-bottom:0.4rem;">
        © 2025 Cláudio Ferreira Neves · Especialista em Ciência de Dados &amp; IA · SENAI Pará · CentroWEG/SC
    </p>
    <p style="font-size:0.82rem;">
        Material didático autoral — reprodução sem crédito ao autor é considerada plágio.
    </p>
</div>
""", unsafe_allow_html=True)
