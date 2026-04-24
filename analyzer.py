import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import numpy as np
import re

# ==========================================
# LIQUID GLASS UI (V11 ALPHA)
# ==========================================
st.set_page_config(page_title="Lewiatan V11 ALPHA", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
    .stApp { background: radial-gradient(circle at 2% 2%, #121217 0%, #050505 50%, #020202 100%) !important; font-family: 'Inter', sans-serif; }
    .main { background-color: transparent !important; }
    div[data-testid="metric-container"] { background: rgba(255, 255, 255, 0.03) !important; backdrop-filter: blur(20px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-top: 1px solid rgba(255, 255, 255, 0.15) !important; border-radius: 16px !important; padding: 24px !important; box-shadow: 0 10px 40px rgba(0,0,0,0.4) !important; transition: all 0.4s ease !important; }
    div[data-testid="metric-container"]:hover { transform: translateY(-5px) scale(1.02); border: 1px solid rgba(0, 122, 255, 0.4) !important; box-shadow: 0 15px 50px rgba(0, 122, 255, 0.15) !important; }
    h1, h2, h3, p, label { color: #EBEBEB !important; font-family: 'Inter', sans-serif !important; }
    div[data-testid="stMetricValue"] > div { font-family: 'JetBrains Mono', monospace !important; font-size: 2.2rem !important; color: #007AFF !important; text-shadow: 0 0 20px rgba(0, 122, 255, 0.3); }
    .stTabs [data-baseweb="tab-list"] { background-color: rgba(255, 255, 255, 0.03); padding: 8px; border-radius: 16px; gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 42px; border-radius: 10px; color: #888; font-weight: 600; border: none; transition: 0.3s; }
    .stTabs [aria-selected="true"] { background-color: #1D1D1F !important; color: #007AFF !important; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    .js-plotly-plot { border-radius: 16px !important; overflow: hidden !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }
    </style>
    """, unsafe_allow_html=True)

if 'selected_asset' not in st.session_state:
    st.session_state['selected_asset'] = None


# ==========================================
# PARSER V11 (ALPHA & FORWARD RETURNS)
# ==========================================
@st.cache_data(show_spinner=False)
def process_log(file_name, raw_bytes):
    raw_str = raw_bytes.decode("utf-8")
    sandbox, activities, trades = "", "", []
    try:
        data = json.loads(raw_str)
        activities = data.get("activitiesLog", "")
        trades = data.get("tradeHistory", [])
        sandbox = data.get("sandboxLog", "")
    except:
        sections = re.split(r'(Sandbox logs:|Activities log:|Trade History:)', raw_str)
        current = ""
        for i in range(len(sections)):
            if sections[i] == "Sandbox logs:":
                current = "SB"
            elif sections[i] == "Activities log:":
                current = "ACT"
            elif sections[i] == "Trade History:":
                current = "TR"
            else:
                if current == "SB":
                    sandbox += sections[i]
                elif current == "ACT":
                    activities += sections[i]
                elif current == "TR":
                    try:
                        m = re.search(r'\[.*\]', sections[i], re.DOTALL)
                        if m: trades.extend(json.loads(m.group()))
                    except:
                        pass

    if activities:
        df = pd.read_csv(io.StringIO(activities.strip()), sep=";")
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(int)

        for col in df.columns:
            if 'volume' in col or col in ['profit_and_loss', 'position']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            elif 'price' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'position' not in df.columns: df['position'] = 0
        if 'profit_and_loss' not in df.columns: df['profit_and_loss'] = 0

        for col in ['bid_volume_2', 'bid_volume_3', 'ask_volume_2', 'ask_volume_3']:
            if col not in df.columns: df[col] = 0

        # Bazowe wskaźniki kwantowe
        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
        df['spread'] = df['ask_price_1'] - df['bid_price_1']

        b_press = df['bid_volume_1'].abs() + (df['bid_volume_2'].abs() * 0.5) + (df['bid_volume_3'].abs() * 0.25)
        a_press = df['ask_volume_1'].abs() + (df['ask_volume_2'].abs() * 0.5) + (df['ask_volume_3'].abs() * 0.25)
        total_p = b_press + a_press
        df['l3_pressure'] = np.where(total_p > 0, (b_press - a_press) / total_p, 0)

        df['inv_stress'] = (df['position'].abs() / 100) * 100
        df['volatility'] = df.groupby('product')['mid_price'].transform(lambda x: x.rolling(20, min_periods=1).std())

        v_total = df['bid_volume_1'].abs() + df['ask_volume_1'].abs()
        df['vwmp'] = np.where(v_total > 0, (df['bid_price_1'] * df['ask_volume_1'].abs() + df['ask_price_1'] * df[
            'bid_volume_1'].abs()) / v_total, df['mid_price'])
        df['toxicity'] = (df['vwmp'] - df['mid_price']).abs()

        # ==========================================
        # NOWOŚĆ V11: SYGNAŁY I FORWARD RETURNS
        # ==========================================
        # 1. Order Flow Imbalance (OFI) - zmiana na L1
        df['bid_diff'] = df.groupby('product')['bid_volume_1'].diff().fillna(0)
        df['ask_diff'] = df.groupby('product')['ask_volume_1'].diff().fillna(0)
        df['ofi'] = df['bid_diff'] - df['ask_diff']

        # 2. Micro-Price Momentum
        df['micro_momentum'] = df.groupby('product')['vwmp'].diff(5).fillna(0)

        # 3. Przyszłe zwroty (Forward Returns) do testowania hipotez
        df['fwd_ret_10'] = df.groupby('product')['mid_price'].shift(-10) - df['mid_price']
        df['fwd_ret_20'] = df.groupby('product')['mid_price'].shift(-20) - df['mid_price']

        return df, trades, sandbox
    return pd.DataFrame(), [], ""


def extract_vars(sandbox):
    return re.findall(r"([A-Z_]+):\s*(-?\d+\.?\d*)", sandbox)


# ==========================================
# RENDEROWANIE DASHBOARDU
# ==========================================
with st.sidebar:
    st.markdown("##  Lewiatan Control")
    files = st.file_uploader("Wgraj .log / .json", accept_multiple_files=True)
    show_trades = st.checkbox("Pokaż zlecenia na wykresach", value=True)
    st.info("Status: Alpha Engine Online.")

if files:
    all_dfs, all_trades, all_sb = [], [], ""
    with st.spinner("Kompilacja macierzy predykcyjnych..."):
        for f in files:
            d, t, s = process_log(f.name, f.getvalue())
            if not d.empty: all_dfs.append(d)
            all_trades.extend(t)
            all_sb += s + "\n"

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True).sort_values(['product', 'timestamp'])
        products = df['product'].dropna().unique().tolist()

        if st.session_state['selected_asset'] not in products:
            st.session_state['selected_asset'] = products[0]

        # 6 ZAKŁADEK (DODANA ALPHA)
        t_global, t_quant, t_dyn, t_l3, t_alpha, t_prompt = st.tabs([
            "🌐 GLOBAL OVERVIEW", "📊 QUANT TERMINAL", "🧬 MARKET DYNAMICS",
            "🏛 L3 ORDER BOOK", "🧪 ALPHA RESEARCH", "🤖 SMART PROMPT"
        ])

        with t_global:
            daily_pnl = df.groupby('timestamp')['profit_and_loss'].sum().reset_index()
            total_pnl = daily_pnl['profit_and_loss'].iloc[-1] if not daily_pnl.empty else 0
            max_dd = (daily_pnl['profit_and_loss'].cummax() - daily_pnl[
                'profit_and_loss']).max() if not daily_pnl.empty else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Global Portfolio Net PnL", f"{total_pnl:,.0f} X")
            c2.metric("Portfolio Max Drawdown", f"-{max_dd:,.0f}", delta_color="inverse")
            c3.metric("Total Executed Trades", len(all_trades))
            c4.metric("Active Tradable Assets", len(products))

            st.markdown("### Global Equity Curve")
            fig_g = go.Figure(go.Scatter(x=daily_pnl['timestamp'], y=daily_pnl['profit_and_loss'], fill='tozeroy',
                                         line=dict(color='#007AFF', width=3)))
            fig_g.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("---")
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            st.session_state['selected_asset'] = st.selectbox("🎯 Wybierz Instrument do analizy szczebla taktycznego:",
                                                              products,
                                                              index=products.index(st.session_state['selected_asset']))

        df_a = df[df['product'] == st.session_state['selected_asset']].copy()
        ts_min, ts_max = int(df_a['timestamp'].min()), int(df_a['timestamp'].max())
        time_range = st.slider("Zakres Czasowy (Timestamp)", min_value=ts_min, max_value=ts_max, value=(ts_min, ts_max),
                               step=100)
        df_filt = df_a[(df_a['timestamp'] >= time_range[0]) & (df_a['timestamp'] <= time_range[1])]

        with t_quant:
            if not df_filt.empty:
                c1, c2, c3, c4 = st.columns(4)
                p_start, p_end = df_filt['profit_and_loss'].iloc[0], df_filt['profit_and_loss'].iloc[-1]
                c1.metric("Local PnL Delta", f"{p_end - p_start:,.0f} X", delta=f"{p_end - p_start:,.0f}")
                c2.metric("Peak Inventory Stress", f"{df_filt['inv_stress'].max():.1f}%")
                c3.metric("Avg Spread", f"{df_filt['spread'].mean():.2f}")
                c4.metric("Avg Book Pressure", f"{df_filt['l3_pressure'].mean():.3f}")

            fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig_q.add_trace(go.Scatter(x=df_filt['timestamp'], y=df_filt['ask_price_1'], name="Ask",
                                       line=dict(color='rgba(255,55,95,0.4)', width=1)), row=1, col=1)
            fig_q.add_trace(go.Scatter(x=df_filt['timestamp'], y=df_filt['bid_price_1'], fill='tonexty',
                                       fillcolor='rgba(255,255,255,0.05)', name="Bid",
                                       line=dict(color='rgba(48,209,88,0.4)', width=1)), row=1, col=1)
            fig_q.add_trace(go.Scatter(x=df_filt['timestamp'], y=df_filt['mid_price'], name="Mid Price",
                                       line=dict(color='#007AFF', width=3)), row=1, col=1)
            fig_q.add_trace(go.Scatter(x=df_filt['timestamp'], y=df_filt['vwmp'], name="VWMP",
                                       line=dict(color='white', dash='dot', width=1)), row=1, col=1)

            if show_trades and all_trades:
                tdf = pd.DataFrame(all_trades)
                if 'symbol' in tdf.columns:
                    tdf = tdf[
                        (tdf['symbol'] == st.session_state['selected_asset']) & (tdf['timestamp'] >= time_range[0]) & (
                                    tdf['timestamp'] <= time_range[1])]
                    buys = tdf[tdf['buyer'] == "SUBMISSION"]
                    sells = tdf[tdf['seller'] == "SUBMISSION"]
                    fig_q.add_trace(go.Scatter(x=buys['timestamp'], y=buys['price'], mode='markers', name='BUY Exec',
                                               marker=dict(symbol='triangle-up', size=14, color='#30D158',
                                                           line=dict(width=1, color='white'))), row=1, col=1)
                    fig_q.add_trace(go.Scatter(x=sells['timestamp'], y=sells['price'], mode='markers', name='SELL Exec',
                                               marker=dict(symbol='triangle-down', size=14, color='#FF375F',
                                                           line=dict(width=1, color='white'))), row=1, col=1)

            fig_q.add_trace(go.Bar(x=df_filt['timestamp'], y=df_filt['position'], name="Inventory",
                                   marker=dict(color=df_filt['inv_stress'], colorscale='Bluered', showscale=False)),
                            row=2, col=1)
            fig_q.update_layout(height=650, template="plotly_dark", hovermode="x unified",
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_q, use_container_width=True)

        with t_dyn:
            c_d1, c_d2, c_d3 = st.columns(3)
            with c_d1:
                f_vol = go.Figure(
                    go.Scatter(x=df_filt['timestamp'], y=df_filt['volatility'], line=dict(color='#BF5AF2', width=2),
                               fill='tozeroy'))
                f_vol.update_layout(title="Realized Volatility (Roll 20)", template="plotly_dark", height=250,
                                    paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(f_vol, use_container_width=True)
            with c_d2:
                f_tox = go.Figure(
                    go.Scatter(x=df_filt['timestamp'], y=df_filt['toxicity'], line=dict(color='#FFD60A', width=2),
                               fill='tozeroy'))
                f_tox.update_layout(title="Order Toxicity Index", template="plotly_dark", height=250,
                                    paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(f_tox, use_container_width=True)
            with c_d3:
                f_spr = go.Figure(
                    go.Scatter(x=df_filt['timestamp'], y=df_filt['spread'], line=dict(color='#FF9F0A', width=2)))
                f_spr.update_layout(title="Market Spread", template="plotly_dark", height=250,
                                    paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(f_spr, use_container_width=True)

        with t_l3:
            f_l3 = go.Figure()
            for i in [3, 2, 1]:
                if f'ask_price_{i}' in df_filt.columns:
                    f_l3.add_trace(go.Scatter(x=df_filt['timestamp'], y=df_filt[f'ask_price_{i}'], name=f"Ask L{i}",
                                              line=dict(width=1, color=f'rgba(255, 55, 95, {0.3 * i})')))
            for i in [1, 2, 3]:
                if f'bid_price_{i}' in df_filt.columns:
                    f_l3.add_trace(go.Scatter(x=df_filt['timestamp'], y=df_filt[f'bid_price_{i}'], name=f"Bid L{i}",
                                              line=dict(width=1, color=f'rgba(48, 209, 88, {0.3 * i})')))
            f_l3.update_layout(title="Wizualizacja głębi arkusza (L1-L3)", template="plotly_dark", height=450,
                               paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(f_l3, use_container_width=True)

        # -------------------------------
        # NOWOŚĆ: ALPHA RESEARCH
        # -------------------------------
        with t_alpha:
            st.markdown(f"### Laboratorium Alfy: `{st.session_state['selected_asset']}`")
            st.markdown("Korelacja wyliczonych sygnałów kwantowych z przyszłymi ruchami cenowymi.")

            c_al1, c_al2 = st.columns([1, 2])

            with c_al1:
                st.markdown("#### Wybierz Sygnał do Zbadania")
                signal_choice = st.selectbox("Sygnał bazowy", ["l3_pressure", "ofi", "micro_momentum", "toxicity"])
                fwd_choice = st.radio("Horyzont czasowy (T+X)", ["fwd_ret_10", "fwd_ret_20"])

                # Obliczenie IC (Information Coefficient)
                df_clean = df_filt[[signal_choice, fwd_choice]].dropna()
                if not df_clean.empty and len(df_clean) > 5:
                    ic_pearson = df_clean[signal_choice].corr(df_clean[fwd_choice], method='pearson')
                    ic_spearman = df_clean[signal_choice].corr(df_clean[fwd_choice], method='spearman')

                    st.metric("Information Coefficient (Pearson)", f"{ic_pearson:.4f}",
                              help="Korelacja liniowa. >0.05 to mocny sygnał w HFT.")
                    st.metric("Rank Coefficient (Spearman)", f"{ic_spearman:.4f}",
                              help="Korelacja rangowa (odporna na wartości skrajne).")
                else:
                    st.warning("Zbyt mało danych do obliczenia korelacji w tym oknie.")

            with c_al2:
                # Scatter Plot z Linią Regresji
                if not df_clean.empty and len(df_clean) > 5:
                    fig_scatter = px.scatter(
                        df_clean, x=signal_choice, y=fwd_choice,
                        trendline="ols", trendline_color_override="#FF375F",
                        title=f"Predykcja: {signal_choice} vs Zmiana ceny za {fwd_choice.split('_')[2]} ticków",
                        template="plotly_dark"
                    )
                    fig_scatter.update_traces(marker=dict(size=4, color='#007AFF', opacity=0.5))
                    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
                    st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")
            st.markdown("### Cross-Asset Lead-Lag Matrix")
            st.markdown("Szukanie zależności pomiędzy różnymi aktywami w portfelu.")

            if len(products) > 1:
                # Pivot do macierzy korelacji
                pivot_df = df.pivot_table(index='timestamp', columns='product', values='mid_price').ffill().dropna()
                rets_df = pivot_df.pct_change().dropna()
                corr_matrix = rets_df.corr().round(3)

                fig_hm = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu', zmid=0,
                    text=corr_matrix.values, texttemplate="%{text}", textfont={"size": 14}
                ))
                fig_hm.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Wgraj plik z wieloma instrumentami, aby wygenerować macierz Cross-Asset.")

                # -------------------------------
                # 6. SMART PROMPT GENERATOR (V11.1 DUAL MODE)
                # -------------------------------
            with t_prompt:
                st.markdown("### 🤖 Generatory Raportów LLM")
                prompt_mode = st.radio("Wybierz tryb analizy:", ["Taktyczna Ocena Egzekucji (T-100)",
                                                                 "Kwantowe Poszukiwanie Alfy (IC & StatArb)"],
                                       horizontal=True)

                c_vars = extract_vars(all_sb)
                var_str = "Zmienne konfiguracyjne z Sandboxa:\n" + "\n".join(
                    [f"- {v[0]}: {v[1]}" for v in c_vars]) if c_vars else "Brak zalogowanych zmiennych customowych."
                ps = df_filt['profit_and_loss'].iloc[0] if not df_filt.empty else 0
                pe = df_filt['profit_and_loss'].iloc[-1] if not df_filt.empty else 0

                if prompt_mode == "Taktyczna Ocena Egzekucji (T-100)":
                    # --- STARY TRYB EGZEKUCJI ---
                    base_cols = ['timestamp', 'bid_price_1', 'ask_price_1', 'mid_price', 'vwmp', 'l3_pressure',
                                 'position']
                    cols_to_show = [c for c in base_cols if c in df_filt.columns]
                    if 'position' in cols_to_show and df_filt['position'].abs().sum() == 0:
                        cols_to_show.remove('position')

                    trade_context = "Brak naszych transakcji w wybranym oknie."
                    anomalies_context = "Brak danych."

                    if not df_filt.empty:
                        if all_trades:
                            tdf = pd.DataFrame(all_trades)
                            if 'symbol' in tdf.columns:
                                tdf = tdf[(tdf['symbol'] == st.session_state['selected_asset']) & (
                                        tdf['timestamp'] >= time_range[0]) & (
                                                  tdf['timestamp'] <= time_range[1])]
                                my_trades = tdf[
                                    (tdf['buyer'] == "SUBMISSION") | (tdf['seller'] == "SUBMISSION")].copy()
                                if not my_trades.empty:
                                    my_trades['ACTION'] = np.where(my_trades['buyer'] == "SUBMISSION", 'BUY',
                                                                   'SELL')
                                    my_trades = my_trades.rename(columns={'price': 'EXEC_PRICE', 'quantity': 'VOL'})

                                    trade_ticks = my_trades['timestamp'].unique().tolist()
                                    context_ticks = set()
                                    for t in trade_ticks:
                                        context_ticks.add(t - 100)
                                        context_ticks.add(t)

                                    df_context = df_filt[df_filt['timestamp'].isin(context_ticks)][
                                        cols_to_show].copy()
                                    merged = pd.merge(df_context,
                                                      my_trades[['timestamp', 'ACTION', 'EXEC_PRICE', 'VOL']],
                                                      on='timestamp', how='left')
                                    merged['ACTION'] = merged['ACTION'].fillna('---')
                                    merged['EXEC_PRICE'] = merged['EXEC_PRICE'].fillna(0).astype(int)
                                    merged['VOL'] = merged['VOL'].fillna(0).astype(int)

                                    if 'l3_pressure' in merged.columns: merged['l3_pressure'] = merged[
                                        'l3_pressure'].round(3)
                                    if 'vwmp' in merged.columns: merged['vwmp'] = merged['vwmp'].round(2)

                                    col_order = ['timestamp', 'ACTION', 'EXEC_PRICE', 'VOL'] + [c for c in
                                                                                                cols_to_show if
                                                                                                c != 'timestamp']
                                    merged = merged[col_order]
                                    trade_context = merged.to_string(index=False)

                        try:
                            max_spread_idx = df_filt['spread'].idxmax()
                            max_tox_idx = df_filt['toxicity'].idxmax()
                            df_anomalies = df_filt.loc[[max_spread_idx, max_tox_idx].copy()][cols_to_show]
                            df_anomalies.insert(0, 'ANOMALIA', ['MAX SPREAD', 'MAX TOXICITY'])
                            if 'l3_pressure' in df_anomalies.columns: df_anomalies['l3_pressure'] = df_anomalies[
                                'l3_pressure'].round(3)
                            if 'vwmp' in df_anomalies.columns: df_anomalies['vwmp'] = df_anomalies['vwmp'].round(2)
                            anomalies_context = df_anomalies.to_string(index=False)
                        except:
                            anomalies_context = "Zbyt mało danych do wyliczenia anomalii."

                    prompt_text = f"""RAPORT TAKTYCZNY: {st.session_state['selected_asset']}
                ZAKRES TICKÓW: {time_range[0]} do {time_range[1]}
                Delta PnL: {pe - ps:,.0f} | Średni Spread: {df_filt['spread'].mean() if not df_filt.empty else 0:.2f}

                {var_str}

                --- STANY ARKUSZA: IMPULS (T-100) vs EGZEKUCJA (T) ---
                {trade_context}

                --- NAJWIĘKSZE ANOMALIE RYNKOWE W TYM OKNIE ---
                {anomalies_context}

                ZADANIE: Przeanalizuj pary ticków (Impuls vs Egzekucja). Oceń, co sprowokowało algorytm do działania. Jak zmienił się VWMP lub l3_pressure między stanem T-100 a momentem zawarcia transakcji? Czy decyzja była optymalna w obliczu tych zmian?"""
                    st.text_area("Skopiuj ten blok tekstu:", value=prompt_text, height=450)

                else:
                    # --- NOWY TRYB: ALPHA DISCOVERY ---
                    st.markdown("System kompresuje macierze IC oraz korelacje Cross-Asset do formatu tekstowego...")

                    # 1. Automatyczne liczenie IC dla wszystkich sygnałów
                    signals = ["l3_pressure", "ofi", "micro_momentum", "toxicity"]
                    targets = ["fwd_ret_10", "fwd_ret_20"]

                    ic_results = []
                    if not df_filt.empty:
                        for sig in signals:
                            for tgt in targets:
                                if sig in df_filt.columns and tgt in df_filt.columns:
                                    df_c = df_filt[[sig, tgt]].dropna()
                                    if len(df_c) > 10:
                                        ic_s = df_c[sig].corr(df_c[tgt], method='spearman')
                                        ic_p = df_c[sig].corr(df_c[tgt], method='pearson')
                                        ic_results.append(
                                            f"- {sig.ljust(15)} -> {tgt.ljust(10)} | Spearman: {ic_s:+.4f} | Pearson: {ic_p:+.4f}")

                    ic_str = "\n".join(
                        ic_results) if ic_results else "Brak wystarczających danych do obliczenia IC."

                    # 2. Liczenie korelacji Cross-Asset
                    cross_asset_str = "Brak innych instrumentów w tym oknie czasowym do wyliczenia macierzy."
                    if len(products) > 1:
                        try:
                            pivot_df = df_filt.pivot_table(index='timestamp', columns='product',
                                                           values='mid_price').ffill().dropna()
                            rets_df = pivot_df.pct_change().dropna()
                            corr_matrix = rets_df.corr().round(3)
                            sel_asset = st.session_state['selected_asset']
                            if sel_asset in corr_matrix.columns:
                                corr_with_others = corr_matrix[sel_asset].drop(sel_asset).to_dict()
                                cross_asset_str = "\n".join(
                                    [f"- Korelacja zwrotów z {k}: {v:+.3f}" for k, v in corr_with_others.items()])
                        except Exception as e:
                            cross_asset_str = f"Błąd wyliczenia macierzy korelacji: {e}"

                    alpha_prompt_text = f"""RAPORT KWANTOWY (ALPHA DISCOVERY): {st.session_state['selected_asset']}
                ZAKRES TICKÓW: {time_range[0]} do {time_range[1]}

                1. WYNIKI TESTÓW PREDYKCYJNYCH (INFORMATION COEFFICIENT):
                (Współczynnik korelacji między sygnałem a przyszłą zmianą ceny. Wartości > 0.05 lub < -0.05 to silne anomalie w HFT)
                {ic_str}

                2. CROSS-ASSET KORELACJE (LEAD-LAG POTENTIAL):
                {cross_asset_str}

                ZADANIE: Jesteś Inżynierem Kwantowym pracującym nad systemem G.O.A.T. Przeanalizuj powyższe wyniki:
                1. Który z sygnałów (L3 Pressure, OFI, Momentum, Toxicity) wykazuje największą moc predykcyjną (najwyższy absolutny współczynnik Spearman/Pearson) dla T+10 i T+20? Co z tego wynika mechanicznie?
                2. Zwróć uwagę na znak korelacji. Czy to sygnał podążania za trendem (Momentum), czy odwrócenia (Mean Reversion)?
                3. Na podstawie korelacji Cross-Asset, czy dostrzegasz potencjał na Arbitraż Statystyczny z innym instrumentem? 
                Zaproponuj logikę wejścia/wyjścia (Entry/Exit criteria) dla nowej strategii opartej na najsilniejszym z tych wektorów."""

                    st.text_area("Skopiuj ten blok tekstu i przekaż do analizy:", value=alpha_prompt_text,
                                 height=450)

                with st.expander("Rozwiń by zobaczyć oryginalne logi print() z Pythona"):
                    st.text(all_sb)

    else:
            st.markdown(
                "<h2 style='text-align: center; margin-top: 20%; color: #888;'>Czekam na przesył danych... Wgraj log z platformy w panelu bocznym.</h2>",
                unsafe_allow_html=True)