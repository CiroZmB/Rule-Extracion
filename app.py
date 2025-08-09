import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
from tabs.utils import *

# ================================================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ================================================================================================

st.set_page_config(
    page_title="Trading Analytics App",
    page_icon="📈",
    layout="wide"
)

# ================================================================================================
# INICIALIZACIÓN DEL ESTADO
# ================================================================================================

def initialize_session_state():
    """Inicializa el estado de la sesión"""
    # Estados de los pasos del pipeline
    if 'step_load' not in st.session_state:
        st.session_state.step_load = False
    if 'step_target' not in st.session_state:
        st.session_state.step_target = False
    if 'step_split' not in st.session_state:
        st.session_state.step_split = False
    if 'step_feature_selection' not in st.session_state:
        st.session_state.step_feature_selection = False
    if 'step_extraction' not in st.session_state:
        st.session_state.step_extraction = False
    if 'step_validation' not in st.session_state:
        st.session_state.step_validation = False
    if 'step_ensemble' not in st.session_state:
        st.session_state.step_ensemble = False
    if 'step_optimization' not in st.session_state:  
        st.session_state.step_optimization = False  
    if 'step_sizing' not in st.session_state:
        st.session_state.step_sizing = False 
    
    # DataFrames del pipeline
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_with_target' not in st.session_state:
        st.session_state.df_with_target = None
    if 'train_df' not in st.session_state:
        st.session_state.train_df = None
    if 'test_df' not in st.session_state:
        st.session_state.test_df = None
    
    # Identificadores para evitar re-procesamiento
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None

# ================================================================================================
# FUNCIONES AUXILIARES PARA GRÁFICOS
# ================================================================================================

def create_price_chart(df, title="Evolución del Precio de Cierre"):
    """Crea un gráfico de línea del precio de cierre"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Close:</b> %{y:.5f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_split_chart(df, train_df, test_df):
    """Crea gráfico del close con colores diferentes para train y test"""
    fig = go.Figure()
    
    # Gráfico del conjunto de entrenamiento
    if len(train_df) > 0:
        fig.add_trace(go.Scatter(
            x=train_df.index,
            y=train_df['Close'],
            mode='lines',
            name='Train',
            line=dict(color='#2E8B57', width=1.5),
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Close:</b> %{y:.5f}<br><b>Set:</b> Train<extra></extra>'
        ))
    
    # Gráfico del conjunto de prueba
    if len(test_df) > 0:
        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df['Close'],
            mode='lines',
            name='Test',
            line=dict(color='#DC143C', width=1.5),
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Close:</b> %{y:.5f}<br><b>Set:</b> Test<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='División Train/Test del Precio de Cierre', x=0.5, font=dict(size=20)),
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# ================================================================================================
# FUNCIÓN PARA ANÁLISIS DE VALORES NO NUMÉRICOS
# ================================================================================================

def analyze_non_numeric_values(df):
    """
    Analiza valores no numéricos en el DataFrame
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a analizar
        
    Retorna:
    --------
    dict : Diccionario con información sobre valores problemáticos
    """
    results = {
        'columns_with_issues': {},
        'total_columns': len(df.columns),
        'problematic_columns': 0
    }
    
    # Columnas a ignorar (no son indicadores técnicos)
    ignore_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    
    for col in df.columns:
        if col in ignore_cols:
            continue
            
        col_info = {
            'total_rows': len(df),
            'non_null_rows': df[col].notna().sum(),
            'null_rows': df[col].isna().sum(),
            'inf_values': 0,
            'neg_inf_values': 0,
            'non_numeric_values': 0,
            'problematic_samples': []
        }
        
        # Verificar si la columna es numérica
        try:
            # Convertir a numérico para detectar problemas
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            # Contar infinitos
            if np.any(np.isinf(numeric_series.dropna())):
                col_info['inf_values'] = np.sum(np.isposinf(numeric_series))
                col_info['neg_inf_values'] = np.sum(np.isneginf(numeric_series))
            
            # Contar valores que no se pudieron convertir a numérico
            non_numeric_mask = df[col].notna() & numeric_series.isna()
            col_info['non_numeric_values'] = non_numeric_mask.sum()
            
            # Obtener muestras de valores problemáticos
            if col_info['non_numeric_values'] > 0:
                problematic_values = df.loc[non_numeric_mask, col].unique()[:5]  # Máximo 5 ejemplos
                col_info['problematic_samples'] = list(problematic_values)
        
        except Exception as e:
            col_info['error'] = str(e)
        
        # Si hay problemas, agregar a los resultados
        total_issues = (col_info['inf_values'] + col_info['neg_inf_values'] + 
                       col_info['non_numeric_values'])
        
        if total_issues > 0:
            results['columns_with_issues'][col] = col_info
            results['problematic_columns'] += 1
    
    return results

# ================================================================================================
# TABS DEL PIPELINE
# ================================================================================================

def load_tab():
    """Tab LOAD - Carga de datos"""
    st.header("📊 LOAD - Carga de Datos")
    st.markdown("---")
    
    # Mostrar información si ya hay datos cargados
    if st.session_state.step_load and st.session_state.df_raw is not None:
        df_clean = st.session_state.df_raw
        
        st.success("✅ Datos cargados correctamente")
        
        # Mostrar información del dataset
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📅 Fecha Inicial", df_clean.index.min().strftime('%Y-%m-%d'))
        with col2:
            st.metric("📅 Fecha Final", df_clean.index.max().strftime('%Y-%m-%d'))
        with col3:
            st.metric("⏰ Timeframe", detect_timeframe(df_clean))
        
        # ================================================================================================
        # NUEVO: ANÁLISIS DE VALORES NO NUMÉRICOS
        # ================================================================================================
        
        st.markdown("---")
        st.markdown("### 🔍 Análisis de Calidad de Datos")
        
        # Botón para analizar valores no numéricos
        if st.button("🔎 Analizar Valores No Numéricos", type="secondary"):
            with st.spinner('Analizando valores problemáticos...'):
                analysis_results = analyze_non_numeric_values(df_clean)
            
            # Mostrar resultados del análisis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Columnas", analysis_results['total_columns'])
            with col2:
                st.metric("⚠️ Columnas con Problemas", analysis_results['problematic_columns'])
            with col3:
                if analysis_results['problematic_columns'] == 0:
                    st.metric("✅ Estado", "Limpio")
                else:
                    st.metric("❌ Estado", "Requiere Limpieza")
            
            # Mostrar detalles de columnas problemáticas
            if analysis_results['problematic_columns'] > 0:
                st.markdown("#### 🚨 Columnas con Valores Problemáticos:")
                
                for col_name, col_info in analysis_results['columns_with_issues'].items():
                    with st.expander(f"🔧 {col_name}", expanded=False):
                        
                        # Métricas de la columna
                        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                        
                        with col_metrics1:
                            st.metric("🔢 Valores Nulos", f"{col_info['null_rows']:,}")
                        with col_metrics2:
                            st.metric("♾️ Valores Infinitos", 
                                    f"{col_info['inf_values'] + col_info['neg_inf_values']:,}")
                        with col_metrics3:
                            st.metric("❌ Valores No Numéricos", f"{col_info['non_numeric_values']:,}")
                        
                        # Mostrar ejemplos de valores problemáticos
                        if col_info['problematic_samples']:
                            st.markdown("**Ejemplos de valores problemáticos:**")
                            for i, sample in enumerate(col_info['problematic_samples'], 1):
                                st.code(f"{i}. {sample}")
                        
                        # Detalles adicionales
                        if col_info['inf_values'] > 0:
                            st.warning(f"⚠️ {col_info['inf_values']} valores +∞ encontrados")
                        if col_info['neg_inf_values'] > 0:
                            st.warning(f"⚠️ {col_info['neg_inf_values']} valores -∞ encontrados")
            else:
                st.success("🎉 ¡Excelente! No se encontraron valores problemáticos en los indicadores técnicos.")
                st.info("💡 Todos los indicadores contienen valores numéricos válidos.")
        
        # ================================================================================================
        # FIN DEL ANÁLISIS DE VALORES NO NUMÉRICOS
        # ================================================================================================
        
        # Gráfico del precio de cierre
        st.markdown("### 📈 Evolución del Precio")
        fig = create_price_chart(df_clean)
        st.plotly_chart(fig, use_container_width=True)
        
        # Botón para cargar nuevos datos
        if st.button("🔄 Cargar Nuevos Datos", type="secondary"):
            # Reset completo del pipeline
            st.session_state.step_load = False
            st.session_state.step_target = False
            st.session_state.step_split = False
            st.session_state.step_feature_selection = False
            st.session_state.step_extraction = False
            st.session_state.step_validation = False
            st.session_state.step_ensemble = False
            st.session_state.df_raw = None
            st.session_state.df_with_target = None
            st.session_state.train_df = None
            st.session_state.test_df = None
            if 'current_target_key' in st.session_state:
                del st.session_state.current_target_key
            if 'current_split_key' in st.session_state:
                del st.session_state.current_split_key
            st.rerun()
        
        return
    
    # Si no hay datos cargados, mostrar el uploader
    uploaded_file = st.file_uploader(
        "Arrastra y suelta tu archivo CSV aquí",
        type=['csv'],
        help="El archivo debe contener columnas: Date, Open, High, Low, Close + indicadores técnicos"
    )
    
    if uploaded_file is not None:
        # Verificar si es un archivo nuevo comparando el nombre
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_id:
            try:
                # Cargar y limpiar datos
                with st.spinner('Cargando y procesando datos...'):
                    df_raw = pd.read_csv(uploaded_file, low_memory=False)
                    df_clean = clean_data(df_raw.copy())
                    
                    # Guardar en session_state
                    st.session_state.df_raw = df_clean
                    st.session_state.current_file_id = file_id
                    st.session_state.step_load = True
                    
                    # Reset de pasos posteriores solo al cargar archivo nuevo
                    st.session_state.step_target = False
                    st.session_state.step_split = False
                    st.session_state.step_feature_selection = False
                    st.session_state.step_extraction = False
                    st.session_state.step_validation = False
                    st.session_state.step_ensemble = False
                    st.session_state.df_with_target = None
                    st.session_state.train_df = None
                    st.session_state.test_df = None
                    if 'current_target_key' in st.session_state:
                        del st.session_state.current_target_key
                    if 'current_split_key' in st.session_state:
                        del st.session_state.current_split_key
                
                st.success("✅ Datos cargados correctamente")
                st.rerun()  # Refrescar para mostrar los datos
                
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")
                st.session_state.step_load = False
    
    else:
        st.info("👆 Carga un archivo CSV para comenzar")
        st.session_state.step_load = False

def target_tab():
    """Tab TARGET - Configuración de target"""
    st.header("🎯 TARGET - Configuración de Variable Objetivo")
    st.markdown("---")
    
    if not st.session_state.step_load:
        st.warning("⚠️ Primero carga los datos en la sección LOAD")
        return
    
    df = st.session_state.df_raw.copy()
    
    # Controles para configurar el target
    col1, col2 = st.columns(2)
    
    with col1:
        target_type = st.selectbox(
            "Tipo de Target:",
            ["Change", "IBS"],
            help="Change: Retorno basado en precios Open. IBS: Internal Bar Strength extendido"
        )
    
    with col2:
        days_ahead = st.number_input(
            "Velas hacia adelante:",
            min_value=1,
            max_value=50,
            value=1,
            help="Número de velas futuras a considerar para el cálculo del target"
        )
    
    # Botón para generar target
    if st.button("🎯 Generar Target", type="primary", use_container_width=True):
        with st.spinner('Generando target...'):
            if target_type == "Change":
                df_with_target = change_target(df.copy(), days_ahead=days_ahead)
            else:  # IBS
                df_with_target = ibs_target(df.copy(), days_ahead=days_ahead)
            
            st.session_state.df_with_target = df_with_target
            st.session_state.current_target_key = f"{target_type}_{days_ahead}"
            st.session_state.step_target = True
            
            # Reset de pasos posteriores
            st.session_state.step_split = False
            st.session_state.step_feature_selection = False
            st.session_state.step_extraction = False
            st.session_state.step_validation = False
            st.session_state.step_ensemble = False
            st.session_state.train_df = None
            st.session_state.test_df = None
            
            st.success("✅ Target generado correctamente")
    
    # Mostrar resultados si ya se generó el target
    if st.session_state.step_target and st.session_state.df_with_target is not None:
        df_with_target = st.session_state.df_with_target
        target_valid = df_with_target['Target'].dropna()
        
        st.markdown("---")
        st.markdown("### 📊 Información del Target")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Registros válidos", f"{len(target_valid):,}")
        with col2:
            st.metric("📈 Target promedio", f"{target_valid.mean():.6f}")
        with col3:
            st.metric("📉 Target std", f"{target_valid.std():.6f}")
        
        # Histograma del target
        st.markdown("### 📊 Distribución del Target")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=target_valid,
            nbinsx=50,
            name='Target',
            opacity=0.7,
            marker_color='skyblue'
        ))
        
        # Línea vertical para el promedio
        mean_val = target_valid.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Promedio: {mean_val:.6f}",
            annotation_position="top right"
        )
        
        target_info = st.session_state.current_target_key.split('_')
        fig.update_layout(
            title=dict(text=f'Distribución del Target ({target_info[0]} - {target_info[1]} velas)', x=0.5),
            xaxis_title='Valor del Target',
            yaxis_title='Frecuencia',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def split_tab():
    """Tab SPLIT - División de datos"""
    st.header("🔄 SPLIT - División de Datos")
    st.markdown("---")
    
    if not st.session_state.step_target:
        st.warning("⚠️ Primero configura el target en la sección TARGET")
        return
    
    df_with_target = st.session_state.df_with_target.copy()
    
    # Controles para configurar la división
    col1, col2 = st.columns(2)
    
    with col1:
        split_mode = st.selectbox(
            "Modo de División:",
            ["classic", "inverted", "free"],
            help="Classic: train=antiguos, test=recientes. Inverted: train=recientes, test=antiguos. Free: período personalizado"
        )
    
    with col2:
        train_ratio = st.slider(
            "Porcentaje de Train:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            disabled=(split_mode == "free"),
            help="Proporción de datos para entrenamiento"
        )
    
    # Controles adicionales para modo 'free'
    train_start, train_end = None, None
    if split_mode == "free":
        st.markdown("**Configuración de período de entrenamiento:**")
        col1, col2 = st.columns(2)
        
        min_date = df_with_target.index.min()
        max_date = df_with_target.index.max()
        
        with col1:
            train_start = st.date_input(
                "Fecha inicio train:",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            train_end = st.date_input(
                "Fecha fin train:",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
    
    # Botón para ejecutar división
    if st.button("🔄 Ejecutar División", type="primary", use_container_width=True):
        try:
            with st.spinner('Dividiendo datos...'):
                if split_mode == "free":
                    train_df, test_df = split_data(
                        df_with_target.copy(),
                        mode=split_mode,
                        train_start=train_start,
                        train_end=train_end
                    )
                else:
                    train_df, test_df = split_data(
                        df_with_target.copy(),
                        mode=split_mode,
                        train_ratio=train_ratio
                    )
                
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df
                st.session_state.current_split_key = f"{split_mode}_{train_ratio}_{train_start}_{train_end}"
                st.session_state.step_split = True
                
                # Reset de pasos posteriores
                st.session_state.step_feature_selection = False
                st.session_state.step_extraction = False
                st.session_state.step_validation = False
                st.session_state.step_ensemble = False
                
                st.success("✅ División ejecutada correctamente")
        
        except Exception as e:
            st.error(f"❌ Error en la división: {str(e)}")
            return
    
    # Mostrar resultados si ya se ejecutó la división
    if st.session_state.step_split and st.session_state.train_df is not None:
        train_df = st.session_state.train_df
        test_df = st.session_state.test_df
        
        st.markdown("---")
        st.markdown("### 📊 Información de la División")
        
        # Mostrar información de la división
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Registros Train", f"{len(train_df):,}")
            if len(train_df) > 0:
                st.caption(f"📅 {train_df.index.min()} → {train_df.index.max()}")
        
        with col2:
            st.metric("🔴 Registros Test", f"{len(test_df):,}")
            if len(test_df) > 0:
                st.caption(f"📅 {test_df.index.min()} → {test_df.index.max()}")
        
        with col3:
            total_records = len(train_df) + len(test_df)
            train_pct = (len(train_df) / total_records * 100) if total_records > 0 else 0
            st.metric("📊 Ratio Train/Test", f"{train_pct:.1f}% / {100-train_pct:.1f}%")
        
        # Gráfico de la división
        st.markdown("### 📈 Visualización Train/Test")
        fig = create_split_chart(df_with_target, train_df, test_df)
        st.plotly_chart(fig, use_container_width=True)

def feature_selection_tab():
    """Tab FEATURE SELECTION - Selección de características"""
    st.header("🎯 FEATURE SELECTION - Selección de Características")
    st.markdown("---")
    
    if not st.session_state.step_split:
        st.warning("⚠️ Completa primero los pasos anteriores del pipeline")
        return
    
    train_df = st.session_state.train_df.copy()
    
    # Verificar que hay features disponibles
    ignore_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Target']
    available_features = [col for col in train_df.columns if col not in ignore_cols]
    
    if len(available_features) == 0:
        st.error("❌ No se encontraron features en el dataset")
        return
    
    st.info(f"📊 Features disponibles: {len(available_features)}")
    
    # ================================================================================================
    # CONTROLES PRINCIPALES
    # ================================================================================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric_type = st.selectbox(
            "📊 Métrica de Análisis:",
            ["None", "pearson", "spearman", "kendall", "mutual_info"],
            help="None: sin ranking por métrica. Otras: ordena features por correlación con target"
        )
    
    with col2:
        apply_correlation_reduction = st.checkbox(
            "🔗 Aplicar reducción por correlación",
            value=True,
            help="Elimina features altamente correlacionadas entre sí"
        )
    
    # Controles condicionales
    col1, col2 = st.columns(2)
    
    with col1:
        correlation_threshold = st.slider(
            "🎯 Umbral de Correlación:",
            min_value=0.75,
            max_value=0.99,
            value=0.95,
            step=0.01,
            disabled=not apply_correlation_reduction,
            help="Features con correlación superior a este umbral serán eliminadas aleatoriamente"
        )
    
    with col2:
        max_features = st.number_input(
            "📈 Máximo Features:",
            min_value=5,
            max_value=300,
            value=100,
            disabled=(metric_type == "None"),
            help="Número máximo de features a seleccionar (solo aplica si se usa métrica)"
        )
    
    # ================================================================================================
    # RESUMEN DE CONFIGURACIÓN
    # ================================================================================================
    
    st.markdown("### ⚙️ Configuración Seleccionada")
    
    config_summary = []
    
    if metric_type != "None":
        config_summary.append(f"🎯 **Métrica**: {metric_type.title()}")
        config_summary.append(f"📊 **Top Features**: {max_features}")
    else:
        config_summary.append("🎯 **Métrica**: Sin ranking por métrica")
    
    if apply_correlation_reduction:
        config_summary.append(f"🔗 **Reducción por correlación**: Sí (umbral: {correlation_threshold})")
    else:
        config_summary.append("🔗 **Reducción por correlación**: No")
    
    # Determinar escenario
    if metric_type != "None" and apply_correlation_reduction:
        scenario = "📋 **Escenario**: Ranking por métrica + Eliminación de correlacionadas"
    elif metric_type != "None" and not apply_correlation_reduction:
        scenario = "📋 **Escenario**: Solo ranking por métrica"
    elif metric_type == "None" and apply_correlation_reduction:
        scenario = "📋 **Escenario**: Solo eliminación de correlacionadas"
    else:
        scenario = "📋 **Escenario**: Sin filtros (todas las features)"
    
    config_summary.append(scenario)
    
    # Mostrar configuración en un contenedor info
    st.info("\n".join(config_summary))
    
    # ================================================================================================
    # BOTÓN EJECUTAR
    # ================================================================================================
    
    if st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
        with st.spinner('Procesando features...'):
            try:
                # Ejecutar análisis según configuración
                if metric_type != "None" and apply_correlation_reduction:
                    # Escenario C: Ranking por métrica + Eliminación de correlacionadas
                    feature_metrics_df = calculate_feature_metrics(train_df, metric=metric_type)
                    selected_features_df = remove_correlated_features(
                        train_df, 
                        feature_metrics_df, 
                        threshold=correlation_threshold,
                        n_features=max_features,
                        random_removal=False
                    )
                    
                elif metric_type != "None" and not apply_correlation_reduction:
                    # Escenario A: Solo ranking por métrica
                    feature_metrics_df = calculate_feature_metrics(train_df, metric=metric_type)
                    selected_features_df = feature_metrics_df.head(max_features)
                    
                elif metric_type == "None" and apply_correlation_reduction:
                    # Escenario B: Solo eliminación de correlacionadas (aleatoria)
                    selected_features_df = remove_correlated_features(
                        train_df,
                        threshold=correlation_threshold,
                        random_removal=True
                    )
                    
                else:
                    # Escenario D: Sin filtros - todas las features
                    selected_features_df = pd.DataFrame(
                        index=available_features,
                        data={'selected': [True] * len(available_features)}
                    )
                
                # Guardar resultados en session state
                st.session_state.selected_features_df = selected_features_df
                st.session_state.current_metric = metric_type
                st.session_state.step_feature_selection = True
                
                # Reset de pasos posteriores
                st.session_state.step_extraction = False
                st.session_state.step_validation = False
                st.session_state.step_ensemble = False
                
                st.success(f"✅ Análisis completado: {len(selected_features_df)} features seleccionadas")
                
                # Mostrar información adicional sobre las features seleccionadas
                st.markdown("---")
                st.markdown("### 📊 Resultados del Análisis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Features Originales", len(available_features))
                
                with col2:
                    st.metric("✅ Features Seleccionadas", len(selected_features_df))
                
                with col3:
                    reduction_pct = (1 - len(selected_features_df) / len(available_features)) * 100
                    st.metric("📉 Reducción", f"{reduction_pct:.1f}%")
                
                # Mostrar lista de features seleccionadas en un expander
                with st.expander("📋 Lista de Features Seleccionadas", expanded=False):
                    if metric_type != "None" and metric_type in selected_features_df.columns:
                        # Mostrar con valores de métrica
                        display_df = selected_features_df.copy()
                        display_df = display_df.reset_index()
                        display_df.columns = ['Feature', f'{metric_type.title()}']
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        # Mostrar solo nombres
                        feature_list = list(selected_features_df.index)
                        for i, feature in enumerate(feature_list, 1):
                            st.write(f"{i}. {feature}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # ================================================================================================
    # MOSTRAR ESTADO SI YA SE COMPLETÓ
    # ================================================================================================
    
    elif hasattr(st.session_state, 'step_feature_selection') and st.session_state.step_feature_selection:
        st.markdown("---")
        st.success("✅ Feature Selection completado")
        
        selected_features_df = st.session_state.selected_features_df
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("✅ Features Seleccionadas", len(selected_features_df))
        
        with col2:
            if hasattr(st.session_state, 'current_metric'):
                st.metric("📊 Métrica Usada", st.session_state.current_metric.title())
            else:
                st.metric("📊 Métrica Usada", "No disponible")
        
        with col3:
            # Calcular reducción basada en features disponibles actuales
            current_available = len([col for col in train_df.columns if col not in ignore_cols])
            reduction_pct = (1 - len(selected_features_df) / current_available) * 100
            st.metric("📉 Reducción", f"{reduction_pct:.1f}%")
        
        # Botón para reconfigurar
        if st.button("🔄 Reconfigurar Feature Selection", type="secondary"):
            st.session_state.step_feature_selection = False
            st.session_state.step_extraction = False
            st.session_state.step_validation = False
            st.session_state.step_ensemble = False
            if hasattr(st.session_state, 'selected_features_df'):
                del st.session_state.selected_features_df
            if hasattr(st.session_state, 'current_metric'):
                del st.session_state.current_metric
            st.rerun()

def extraction_tab():
    """Tab EXTRACTION - Extracción de características"""
    st.header("📦 EXTRACTION - Extracción de Características")
    st.markdown("---")
    
    if not st.session_state.step_feature_selection:
        st.warning("⚠️ Completa primero la selección de características")
        return
    
    # Obtener datos necesarios
    train_df = st.session_state.train_df.copy()
    test_df = st.session_state.test_df.copy()
    selected_features_df = st.session_state.selected_features_df
    
    # Lista de features seleccionadas
    selected_features = list(selected_features_df.index)
    
    st.success(f"✅ {len(selected_features)} features seleccionadas disponibles para extracción")
    
    # ================================================================================================
    # CONFIGURACIÓN DE PARÁMETROS ONER
    # ================================================================================================
    
    st.markdown("### ⚙️ Configuración OneR")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_bins = st.slider(
            "📊 Número de Bins:",
            min_value=3,
            max_value=10,
            value=5,
            help="Número de bins para dividir cada variable"
        )
    
    with col2:
        st.metric("🎯 Variables a Procesar", len(selected_features))
    
    # Información sobre lo que hará el proceso
    st.info(
        f"🔄 Se extraerán **2 reglas por variable** (LONG y SHORT) usando OneR con {n_bins} bins.\n\n"
        f"📊 **Total de reglas esperadas**: {len(selected_features) * 2}\n\n"
        f"📈 **LONG**: Regla con mayor rendimiento positivo\n\n"
        f"📉 **SHORT**: Regla con mayor rendimiento negativo"
    )
    
    # ================================================================================================
    # BOTÓN EJECUTAR EXTRACCIÓN
    # ================================================================================================
    
    if st.button("🚀 Ejecutar Extracción OneR", type="primary", use_container_width=True):
        
        with st.spinner('Extrayendo reglas OneR...'):
            try:
                # Ejecutar extracción
                reglas_df = extraer_mejores_reglas(
                    train_df, 
                    test_df, 
                    selected_features, 
                    n_bins=n_bins
                )
                
                # Guardar resultados
                st.session_state.reglas_extraidas_df = reglas_df
                st.session_state.current_n_bins = n_bins
                st.session_state.step_extraction = True
                
                # Reset de pasos posteriores
                st.session_state.step_validation = False
                st.session_state.step_ensemble = False
                
                st.success(f"✅ Extracción completada: {len(reglas_df)} reglas generadas")
                
            except Exception as e:
                st.error(f"❌ Error en la extracción: {str(e)}")
                return
    
    # ================================================================================================
    # MOSTRAR RESULTADOS SI YA SE COMPLETÓ
    # ================================================================================================
    
    if st.session_state.step_extraction and 'reglas_extraidas_df' in st.session_state:
        
        reglas_df = st.session_state.reglas_extraidas_df
        
        st.markdown("---")
        st.markdown("### 📊 Resultados de la Extracción")
        
        if len(reglas_df) == 0:
            st.warning("⚠️ No se generaron reglas. Verifica la calidad de los datos.")
            return
        
        # ================================================================================================
        # MÉTRICAS GENERALES
        # ================================================================================================
        
        # Analizar performance
        stats = analizar_performance_reglas(reglas_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Total Reglas", stats['total_reglas'])
        
        with col2:
            st.metric("📈 Reglas LONG", stats['reglas_long'])
        
        with col3:
            st.metric("📉 Reglas SHORT", stats['reglas_short'])
            
        with col4:
            consistency = stats.get('consistencia_general', 0)
            if pd.isna(consistency):
                consistency = 0
            st.metric("🔄 Consistencia", f"{consistency:.3f}")
        
        # ================================================================================================
        # PERFORMANCE TRAIN VS TEST
        # ================================================================================================
        
        st.markdown("### 📈 Performance Train vs Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏋️ Performance en TRAIN**")
            train_col1, train_col2 = st.columns(2)
            
            with train_col1:
                st.metric("📊 Promedio General", f"{stats['train_rendimiento_promedio']:.6f}")
                st.metric("📈 Promedio LONG", f"{stats['train_rendimiento_long_promedio']:.6f}")
            
            with train_col2:
                st.metric("📉 Promedio SHORT", f"{stats['train_rendimiento_short_promedio']:.6f}")
        
        with col2:
            st.markdown("**🧪 Performance en TEST**")
            test_col1, test_col2 = st.columns(2)
            
            with test_col1:
                st.metric("📊 Promedio General", f"{stats['test_rendimiento_promedio']:.6f}")
                st.metric("📈 Promedio LONG", f"{stats['test_rendimiento_long_promedio']:.6f}")
            
            with test_col2:
                st.metric("📉 Promedio SHORT", f"{stats['test_rendimiento_short_promedio']:.6f}")
        
        # ================================================================================================
        # TABLA DE REGLAS (CON FILTRO DE RENDIMIENTOS POSITIVOS)
        # ================================================================================================
        
        st.markdown("### 📋 Tabla de Reglas Extraídas")
        
        # Filtrar reglas con rendimientos positivos
        reglas_positivas = reglas_df[
            (reglas_df['rendimiento_train'] > 0) & (reglas_df['rendimiento_test'] > 0)
        ].copy()
        
        if len(reglas_positivas) > 0:
            # Preparar DataFrame para mostrar
            display_df = reglas_positivas.copy()
            display_df['rendimiento_train'] = display_df['rendimiento_train'].round(6)
            display_df['rendimiento_test'] = display_df['rendimiento_test'].round(6)
            
            # Ordenar por rendimiento en train (descendente)
            display_df = display_df.sort_values('rendimiento_train', ascending=False)
            
            # Mostrar información del filtrado
            reglas_eliminadas = len(reglas_df) - len(reglas_positivas)
            if reglas_eliminadas > 0:
                st.info(f"🔍 **Filtro aplicado**: Se eliminaron {reglas_eliminadas} reglas con rendimientos negativos. "
                       f"Mostrando {len(reglas_positivas)} reglas con rendimientos positivos en ambos períodos.")
            else:
                st.success(f"✅ Todas las {len(reglas_positivas)} reglas tienen rendimientos positivos.")
            
            # Mostrar tabla
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        else:
            st.error("❌ No hay reglas con rendimientos positivos en ambos períodos (train y test).")
            st.warning("💡 Considera revisar la configuración de OneR o la calidad de los datos.")
        
        # ================================================================================================
        # ANÁLISIS DETALLADO
        # ================================================================================================
        
        st.markdown("### 🔍 Análisis Detallado")
        
        # Tabs para diferentes análisis
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "🏆 Top Reglas", 
            "📊 Distribuciones", 
            "🔄 Consistencia"
        ])
        
        with analysis_tab1:
            st.markdown("#### 🏆 Mejores Reglas por Performance")
            
            # Top 5 por train
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏋️ Top 5 en TRAIN**")
                top_train = display_df.nlargest(5, 'rendimiento_train')[['variable', 'tipo', 'rendimiento_train']]
                for i, (_, row) in enumerate(top_train.iterrows(), 1):
                    st.write(f"{i}. **{row['variable']}** ({row['tipo']}) - {row['rendimiento_train']:.6f}")
            
            with col2:
                st.markdown("**🧪 Top 5 en TEST**")
                top_test = display_df.nlargest(5, 'rendimiento_test')[['variable', 'tipo', 'rendimiento_test']]
                for i, (_, row) in enumerate(top_test.iterrows(), 1):
                    st.write(f"{i}. **{row['variable']}** ({row['tipo']}) - {row['rendimiento_test']:.6f}")
        
        with analysis_tab2:
            st.markdown("#### 📊 Distribución de Rendimientos")
            
            # Selector para tipo de visualización
            dist_option = st.selectbox(
                "🎯 Seleccionar Vista:",
                ["Agregado (TRAIN vs TEST)", "Por Tipo (LONG vs SHORT)", "Detallado (Todo Separado)"]
            )
            
            # Crear histogramas según la opción seleccionada
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Usar reglas con rendimientos positivos para las distribuciones
            reglas_for_dist = reglas_positivas if len(reglas_positivas) > 0 else reglas_df
            
            if dist_option == "Agregado (TRAIN vs TEST)":
                # Vista original mejorada con promedios
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Rendimiento TRAIN', 'Rendimiento TEST')
                )
                
                # Histograma TRAIN
                fig.add_trace(
                    go.Histogram(
                        x=reglas_for_dist['rendimiento_train'],
                        nbinsx=25,
                        name='TRAIN',
                        marker_color='skyblue',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Línea vertical promedio TRAIN
                mean_train = reglas_for_dist['rendimiento_train'].mean()
                fig.add_vline(
                    x=mean_train,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"μ = {mean_train:.4f}",
                    annotation_position="top",
                    row=1, col=1
                )
                
                # Histograma TEST
                fig.add_trace(
                    go.Histogram(
                        x=reglas_for_dist['rendimiento_test'],
                        nbinsx=25,
                        name='TEST',
                        marker_color='lightcoral',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                # Línea vertical promedio TEST
                mean_test = reglas_for_dist['rendimiento_test'].mean()
                fig.add_vline(
                    x=mean_test,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"μ = {mean_test:.4f}",
                    annotation_position="top",
                    row=1, col=2
                )
                
                title_suffix = f" (Solo rendimientos positivos: {len(reglas_for_dist)})" if len(reglas_positivas) > 0 and len(reglas_positivas) < len(reglas_df) else ""
                fig.update_layout(
                    title_text=f"Distribución de Rendimientos: TRAIN vs TEST{title_suffix}",
                    showlegend=False,
                    height=450
                )
                
            elif dist_option == "Por Tipo (LONG vs SHORT)":
                # Separar por tipo LONG/SHORT
                long_rules = reglas_for_dist[reglas_for_dist['tipo'] == 'LONG']
                short_rules = reglas_for_dist[reglas_for_dist['tipo'] == 'SHORT']
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Reglas LONG', 'Reglas SHORT')
                )
                
                # Histograma LONG (Train + Test)
                fig.add_trace(
                    go.Histogram(
                        x=long_rules['rendimiento_train'],
                        nbinsx=20,
                        name='LONG Train',
                        marker_color='green',
                        opacity=0.6
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=long_rules['rendimiento_test'],
                        nbinsx=20,
                        name='LONG Test',
                        marker_color='lightgreen',
                        opacity=0.6
                    ),
                    row=1, col=1
                )
                
                # Promedio LONG
                mean_long = pd.concat([long_rules['rendimiento_train'], long_rules['rendimiento_test']]).mean()
                fig.add_vline(
                    x=mean_long,
                    line_dash="dash",
                    line_color="darkgreen",
                    annotation_text=f"μ = {mean_long:.4f}",
                    annotation_position="top",
                    row=1, col=1
                )
                
                # Histograma SHORT (Train + Test)
                fig.add_trace(
                    go.Histogram(
                        x=short_rules['rendimiento_train'],
                        nbinsx=20,
                        name='SHORT Train',
                        marker_color='red',
                        opacity=0.6
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=short_rules['rendimiento_test'],
                        nbinsx=20,
                        name='SHORT Test',
                        marker_color='lightcoral',
                        opacity=0.6
                    ),
                    row=1, col=2
                )
                
                # Promedio SHORT
                mean_short = pd.concat([short_rules['rendimiento_train'], short_rules['rendimiento_test']]).mean()
                fig.add_vline(
                    x=mean_short,
                    line_dash="dash",
                    line_color="darkred",
                    annotation_text=f"μ = {mean_short:.4f}",
                    annotation_position="top",
                    row=1, col=2
                )
                
                fig.update_layout(
                    title_text="Distribución por Tipo: LONG vs SHORT",
                    height=450,
                    barmode='overlay'
                )
                
            else:  # "Detallado (Todo Separado)"
                # Vista completa con 4 subplots
                long_rules = reglas_for_dist[reglas_for_dist['tipo'] == 'LONG']
                short_rules = reglas_for_dist[reglas_for_dist['tipo'] == 'SHORT']
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'LONG - TRAIN', 'LONG - TEST',
                        'SHORT - TRAIN', 'SHORT - TEST'
                    )
                )
                
                # LONG TRAIN
                fig.add_trace(
                    go.Histogram(
                        x=long_rules['rendimiento_train'],
                        nbinsx=15,
                        name='LONG Train',
                        marker_color='green',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                mean_long_train = long_rules['rendimiento_train'].mean()
                fig.add_vline(
                    x=mean_long_train,
                    line_dash="dash",
                    line_color="darkgreen",
                    annotation_text=f"μ = {mean_long_train:.4f}",
                    annotation_position="top",
                    row=1, col=1
                )
                
                # LONG TEST
                fig.add_trace(
                    go.Histogram(
                        x=long_rules['rendimiento_test'],
                        nbinsx=15,
                        name='LONG Test',
                        marker_color='lightgreen',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                mean_long_test = long_rules['rendimiento_test'].mean()
                fig.add_vline(
                    x=mean_long_test,
                    line_dash="dash",
                    line_color="darkgreen",
                    annotation_text=f"μ = {mean_long_test:.4f}",
                    annotation_position="top",
                    row=1, col=2
                )
                
                # SHORT TRAIN
                fig.add_trace(
                    go.Histogram(
                        x=short_rules['rendimiento_train'],
                        nbinsx=15,
                        name='SHORT Train',
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                mean_short_train = short_rules['rendimiento_train'].mean()
                fig.add_vline(
                    x=mean_short_train,
                    line_dash="dash",
                    line_color="darkred",
                    annotation_text=f"μ = {mean_short_train:.4f}",
                    annotation_position="top",
                    row=2, col=1
                )
                
                # SHORT TEST
                fig.add_trace(
                    go.Histogram(
                        x=short_rules['rendimiento_test'],
                        nbinsx=15,
                        name='SHORT Test',
                        marker_color='lightcoral',
                        opacity=0.7
                    ),
                    row=2, col=2
                )
                mean_short_test = short_rules['rendimiento_test'].mean()
                fig.add_vline(
                    x=mean_short_test,
                    line_dash="dash",
                    line_color="darkred",
                    annotation_text=f"μ = {mean_short_test:.4f}",
                    annotation_position="top",
                    row=2, col=2
                )
                
                fig.update_layout(
                    title_text="Distribución Detallada: LONG/SHORT x TRAIN/TEST",
                    showlegend=False,
                    height=600
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tab3:
            st.markdown("#### 🔄 Consistencia Train vs Test")
            
            # Usar solo reglas con rendimientos positivos
            if len(reglas_positivas) > 0:
                # Normalizar valores entre 0 y 1 para mejor visualización
                train_values = reglas_positivas['rendimiento_train'].values
                test_values = reglas_positivas['rendimiento_test'].values
                
                # Encontrar rango global para normalización consistente
                min_val = min(train_values.min(), test_values.min())
                max_val = max(train_values.max(), test_values.max())
                
                # Evitar división por cero
                if max_val - min_val > 0:
                    train_normalized = (train_values - min_val) / (max_val - min_val)
                    test_normalized = (test_values - min_val) / (max_val - min_val)
                else:
                    train_normalized = np.ones_like(train_values) * 0.5
                    test_normalized = np.ones_like(test_values) * 0.5
                
                # Scatter plot train vs test normalizado
                fig = go.Figure()
                
                # Separar por tipo
                long_rules_pos = reglas_positivas[reglas_positivas['tipo'] == 'LONG']
                short_rules_pos = reglas_positivas[reglas_positivas['tipo'] == 'SHORT']
                
                # LONG rules (normalizadas)
                if len(long_rules_pos) > 0:
                    long_indices = reglas_positivas.index.isin(long_rules_pos.index)
                    fig.add_trace(go.Scatter(
                        x=train_normalized[long_indices],
                        y=test_normalized[long_indices],
                        mode='markers',
                        name='LONG',
                        marker=dict(color='green', size=8),
                        text=long_rules_pos['variable'].values,
                        customdata=np.column_stack([
                            long_rules_pos['rendimiento_train'].values,
                            long_rules_pos['rendimiento_test'].values
                        ]),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Train (norm): %{x:.3f}<br>' +
                                    'Test (norm): %{y:.3f}<br>' +
                                    'Train (real): %{customdata[0]:.6f}<br>' +
                                    'Test (real): %{customdata[1]:.6f}<extra></extra>'
                    ))
                
                # SHORT rules (normalizadas)
                if len(short_rules_pos) > 0:
                    short_indices = reglas_positivas.index.isin(short_rules_pos.index)
                    fig.add_trace(go.Scatter(
                        x=train_normalized[short_indices],
                        y=test_normalized[short_indices],
                        mode='markers',
                        name='SHORT',
                        marker=dict(color='red', size=8),
                        text=short_rules_pos['variable'].values,
                        customdata=np.column_stack([
                            short_rules_pos['rendimiento_train'].values,
                            short_rules_pos['rendimiento_test'].values
                        ]),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Train (norm): %{x:.3f}<br>' +
                                    'Test (norm): %{y:.3f}<br>' +
                                    'Train (real): %{customdata[0]:.6f}<br>' +
                                    'Test (real): %{customdata[1]:.6f}<extra></extra>'
                    ))
                
                # Línea diagonal de referencia (consistencia perfecta)
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Consistencia Perfecta',
                    line=dict(color='gray', dash='dash', width=2),
                    showlegend=True,
                    hovertemplate='Línea de consistencia perfecta<extra></extra>'
                ))
                
                # Calcular correlación con valores normalizados
                if len(reglas_positivas) > 1:
                    correlation = np.corrcoef(train_normalized, test_normalized)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0
                else:
                    correlation = 0
                
                fig.update_layout(
                    title=f'Consistencia Train vs Test (Normalizado)<br><sub>Correlación: {correlation:.3f} | Reglas con rendimientos positivos: {len(reglas_positivas)}</sub>',
                    xaxis_title='Rendimiento TRAIN (Normalizado 0-1)',
                    yaxis_title='Rendimiento TEST (Normalizado 0-1)',
                    height=500,
                    xaxis=dict(range=[0, 1], dtick=0.2),
                    yaxis=dict(range=[0, 1], dtick=0.2)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Información adicional sobre la normalización
                st.info(f"📊 **Normalización aplicada**: Valores escalados entre 0 y 1 para mejor visualización.\n\n"
                       f"📈 **Rango original**: [{min_val:.6f}, {max_val:.6f}]\n\n"
                       f"🎯 **Interpretación**: Puntos cerca de la línea diagonal indican mayor consistencia entre train y test.")
                
                # Métricas de consistencia
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🔗 Correlación", f"{correlation:.3f}")
                
                with col2:
                    # Calcular cuántas reglas están "cerca" de la línea diagonal (diferencia < 0.2 en escala normalizada)
                    differences = np.abs(train_normalized - test_normalized)
                    consistent_rules = np.sum(differences < 0.2)
                    consistency_rate = consistent_rules / len(reglas_positivas) * 100
                    st.metric("📏 Reglas Consistentes", f"{consistent_rules}/{len(reglas_positivas)} ({consistency_rate:.1f}%)")
                
                with col3:
                    # Diferencia promedio normalizada
                    avg_difference = np.mean(differences)
                    st.metric("📊 Diferencia Promedio", f"{avg_difference:.3f}")
            
            else:
                st.warning("⚠️ No hay reglas con rendimientos positivos para analizar consistencia.")
        
        # ================================================================================================
        # BOTÓN PARA RECONFIGURAR
        # ================================================================================================
        
        st.markdown("---")
        if st.button("🔄 Reconfigurar Extracción", type="secondary"):
            st.session_state.step_extraction = False
            st.session_state.step_validation = False
            st.session_state.step_ensemble = False
            if 'reglas_extraidas_df' in st.session_state:
                del st.session_state.reglas_extraidas_df
            if 'current_n_bins' in st.session_state:
                del st.session_state.current_n_bins
            st.rerun()

def validation_tab():
    """Tab VALIDATION - Validación del modelo con combinaciones de reglas"""
    st.header("✔️ VALIDATION - Validación del Modelo")
    st.markdown("---")
    
    if not st.session_state.step_extraction:
        st.warning("⚠️ Completa primero la extracción de características")
        return
    
    # Verificar que hay reglas extraídas
    if 'reglas_extraidas_df' not in st.session_state or len(st.session_state.reglas_extraidas_df) == 0:
        st.error("❌ No hay reglas extraídas para validar")
        return
    
    # Obtener datos necesarios
    train_df = st.session_state.train_df.copy()
    test_df = st.session_state.test_df.copy()
    reglas_df = st.session_state.reglas_extraidas_df.copy()
    
    # Filtrar reglas con rendimientos positivos (del tab anterior)
    reglas_positivas = reglas_df[
        (reglas_df['rendimiento_train'] > 0) & (reglas_df['rendimiento_test'] > 0)
    ].copy()
    
    if len(reglas_positivas) == 0:
        st.error("❌ No hay reglas con rendimientos positivos para validar")
        st.info("💡 Vuelve al tab EXTRACTION para revisar las reglas generadas.")
        return
    
    # Información inicial
    long_rules = reglas_positivas[reglas_positivas['tipo'] == 'LONG']
    short_rules = reglas_positivas[reglas_positivas['tipo'] == 'SHORT']
    
    st.success(f"✅ Reglas con rendimientos positivos: {len(long_rules)} LONG + {len(short_rules)} SHORT = {len(reglas_positivas)} total")
    
    # ================================================================================================
    # INFORMACIÓN SOBRE COMBINACIONES
    # ================================================================================================
    
    # Calcular número de combinaciones posibles
    from math import comb
    
    n_long_combos = comb(len(long_rules), 2) if len(long_rules) >= 2 else 0
    n_short_combos = comb(len(short_rules), 2) if len(short_rules) >= 2 else 0
    total_combos = n_long_combos + n_short_combos
    
    st.markdown("### 🔗 Combinaciones de Reglas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Combinaciones LONG", n_long_combos)
    
    with col2:
        st.metric("📉 Combinaciones SHORT", n_short_combos)
    
    with col3:
        st.metric("🔢 Total Combinaciones", total_combos)
    
    with col4:
        # Calcular tamaño del dataset completo
        df_completo = pd.concat([train_df, test_df], ignore_index=False).sort_index()
        dataset_size = len(df_completo.dropna(subset=['Target']))
        st.metric("📊 Dataset Completo", f"{dataset_size:,}")
    
    if total_combos == 0:
        st.warning("⚠️ No hay suficientes reglas para generar combinaciones (mínimo 2 reglas del mismo tipo).")
        return
    
    # ================================================================================================
    # CONFIGURACIÓN DEL MONKEY TEST
    # ================================================================================================
    
    st.markdown("### ⚙️ Configuración Monkey Test")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fraction = st.slider(
            "📊 Fracción del Dataset:",
            min_value=0.1,
            max_value=1.0,
            value=1/3,
            step=0.05,
            help="Fracción del dataset a usar en cada simulación"
        )
    
    with col2:
        n_simulations = st.selectbox(
            "🔄 Número de Simulaciones:",
            [500, 1000, 2000, 5000],
            index=1,
            help="Más simulaciones = mayor precisión pero más tiempo"
        )
    
    with col3:
        quantile = st.slider(
            "🎯 Quantil de Umbral:",
            min_value=70,
            max_value=95,
            value=80,
            step=5,
            help="Percentil para establecer el umbral de validación"
        )
    
    # Estimación de tiempo
    tiempo_estimado = max(1, total_combos // 2000)  # Aproximación optimista
    
    st.info(
        f"🔄 **Proceso completo**: \n\n"
        f"1️⃣ Generar **{total_combos:,} combinaciones** de 2 reglas cada una\n\n"
        f"2️⃣ Ejecutar **Monkey Test** con {n_simulations:,} simulaciones (quantil {quantile}%)\n\n"
        f"3️⃣ Filtrar combinaciones que **superen los umbrales**\n\n"
        f"4️⃣ Mostrar tabla final con **combinaciones validadas**\n\n"
        f"⏱️ **Tiempo estimado**: ~{tiempo_estimado}-{tiempo_estimado*2} minutos"
    )
    
    # ================================================================================================
    # BOTÓN EJECUTAR COMBINACIONES
    # ================================================================================================
    
    if st.button("🚀 Generar Combinaciones y Ejecutar Validación", type="primary", use_container_width=True):
        
        try:
            with st.spinner('Paso 1/4: Preparando dataset completo...'):
                # Crear dataset completo
                df_completo = pd.concat([train_df, test_df], ignore_index=False).sort_index()
                
            with st.spinner('Paso 2/4: Generando combinaciones...'):
                # Generar combinaciones
                combinaciones = generar_combinaciones_reglas(reglas_positivas)
                
                if len(combinaciones) == 0:
                    st.error("❌ No se pudieron generar combinaciones")
                    return
                
                st.success(f"✅ {len(combinaciones)} combinaciones generadas")
                
            with st.spinner('Paso 3/4: Evaluando combinaciones...'):
                # Crear contenedor para la barra de progreso
                progress_container = st.container()
                with progress_container:
                    st.write("🔄 Evaluando combinaciones...")
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                # Función callback para actualizar progreso
                def actualizar_progreso(porcentaje):
                    progress_bar.progress(porcentaje / 100)
                    progress_text.text(f"Progreso: {porcentaje}% ({porcentaje * len(combinaciones) // 100}/{len(combinaciones)} combinaciones)")
                
                # Calcular métricas para todas las combinaciones con progreso
                df_combinaciones = calcular_metricas_combinaciones(
                    df_completo, 
                    combinaciones, 
                    progress_callback=actualizar_progreso
                )
                
                # Limpiar la barra de progreso
                progress_container.empty()
                st.success(f"✅ {len(combinaciones)} combinaciones evaluadas")
                
            with st.spinner('Paso 4/4: Ejecutando Monkey Test y validación...'):
                # Ejecutar Monkey Test en dataset completo
                stats_monkey_long = monkey_test_trading(
                    df_completo, 
                    tipo_regla='LONG', 
                    fraction=fraction, 
                    n_simulations=n_simulations, 
                    quantile=quantile
                )
                
                stats_monkey_short = monkey_test_trading(
                    df_completo, 
                    tipo_regla='SHORT', 
                    fraction=fraction, 
                    n_simulations=n_simulations, 
                    quantile=quantile
                )
                
                # Filtrar combinaciones que superan el Monkey Test
                combinaciones_validadas = df_combinaciones.copy()
                
                # Añadir umbrales según tipo
                combinaciones_validadas['umbral_monkey'] = combinaciones_validadas['tipo'].map({
                    'LONG': stats_monkey_long['quantile_value'],
                    'SHORT': stats_monkey_short['quantile_value']
                })
                
                # Filtrar solo las que superan el umbral
                combinaciones_validadas['supera_monkey'] = (
                    combinaciones_validadas['rendimiento_total'] > combinaciones_validadas['umbral_monkey']
                )
                
                combinaciones_finales = combinaciones_validadas[
                    combinaciones_validadas['supera_monkey']
                ].copy()
                
                # Calcular margen de mejora vs monkey test
                if len(combinaciones_finales) > 0:
                    combinaciones_finales['margen_vs_monkey'] = (
                        combinaciones_finales['rendimiento_total'] - combinaciones_finales['umbral_monkey']
                    )
                    
                    combinaciones_finales['factor_mejora'] = combinaciones_finales.apply(
                        lambda row: row['rendimiento_total'] / row['umbral_monkey'] 
                        if row['umbral_monkey'] != 0 else float('inf'), axis=1
                    )
            
            # Guardar resultados en session state
            st.session_state.df_combinaciones_todas = df_combinaciones
            st.session_state.df_combinaciones_validadas = combinaciones_finales
            st.session_state.stats_monkey_long_combo = stats_monkey_long
            st.session_state.stats_monkey_short_combo = stats_monkey_short
            st.session_state.config_validacion_combo = {
                'fraction': fraction,
                'n_simulations': n_simulations,
                'quantile': quantile
            }
            st.session_state.df_completo_validacion = df_completo
            st.session_state.step_validation = True
            
            # Reset pasos posteriores
            st.session_state.step_ensemble = False
            
            st.success(f"✅ Validación completada: {len(combinaciones_finales)} combinaciones validadas de {len(df_combinaciones)} generadas")
            
        except Exception as e:
            st.error(f"❌ Error en el proceso: {str(e)}")
            return
    
    # ================================================================================================
    # MOSTRAR RESULTADOS SI YA SE COMPLETÓ
    # ================================================================================================
    
    if st.session_state.step_validation and 'df_combinaciones_validadas' in st.session_state:
        
        df_combinaciones_todas = st.session_state.df_combinaciones_todas
        df_combinaciones_validadas = st.session_state.df_combinaciones_validadas
        stats_monkey_long = st.session_state.stats_monkey_long_combo
        stats_monkey_short = st.session_state.stats_monkey_short_combo
        config_validacion = st.session_state.config_validacion_combo
        
        st.markdown("---")
        st.markdown("### 📊 Resultados de Validación con Monkey Test")
        
        # ================================================================================================
        # MÉTRICAS DE VALIDACIÓN
        # ================================================================================================
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔢 Combinaciones Generadas", len(df_combinaciones_todas))
        
        with col2:
            st.metric("✅ Combinaciones Validadas", len(df_combinaciones_validadas))
        
        with col3:
            tasa_aprobacion = (len(df_combinaciones_validadas) / len(df_combinaciones_todas) * 100) if len(df_combinaciones_todas) > 0 else 0
            st.metric("📈 Tasa de Aprobación", f"{tasa_aprobacion:.1f}%")
        
        with col4:
            if len(df_combinaciones_validadas) > 0:
                rendimiento_promedio = df_combinaciones_validadas['rendimiento_total'].mean()
                st.metric("💰 Rendimiento Promedio", f"{rendimiento_promedio:.4f}")
            else:
                st.metric("💰 Rendimiento Promedio", "N/A")
        
        # ================================================================================================
        # INFORMACIÓN DE UMBRALES MONKEY TEST
        # ================================================================================================
        
        st.markdown("### 🎯 Umbrales Monkey Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Umbral LONG:**")
            st.metric("🎯 Quantil " + str(config_validacion['quantile']) + "%", f"{stats_monkey_long['quantile_value']:.6f}")
            long_validadas = df_combinaciones_validadas[df_combinaciones_validadas['tipo'] == 'LONG']
            st.write(f"📊 Combinaciones LONG validadas: {len(long_validadas)}")
        
        with col2:
            st.markdown("**📉 Umbral SHORT:**")
            st.metric("🎯 Quantil " + str(config_validacion['quantile']) + "%", f"{stats_monkey_short['quantile_value']:.6f}")
            short_validadas = df_combinaciones_validadas[df_combinaciones_validadas['tipo'] == 'SHORT']
            st.write(f"📊 Combinaciones SHORT validadas: {len(short_validadas)}")
        
        # ================================================================================================
        # TABLA PRINCIPAL - COMBINACIONES VALIDADAS
        # ================================================================================================
        
        st.markdown("### 🏆 Combinaciones que Superan el Monkey Test")
        
        if len(df_combinaciones_validadas) > 0:
            
            # Filtros adicionales para la tabla
            col1, col2 = st.columns(2)
            
            with col1:
                filtro_tipo_validadas = st.selectbox(
                    "🎯 Filtrar por Tipo:",
                    ["Todos", "LONG", "SHORT"],
                    key="filtro_validadas"
                )
            
            with col2:
                min_operaciones_validadas = st.number_input(
                    "📊 Mínimo operaciones:",
                    min_value=0,
                    value=1,
                    key="min_ops_validadas",
                    help="Filtrar combinaciones con al menos N operaciones"
                )
            
            # Aplicar filtros
            df_mostrar = df_combinaciones_validadas.copy()
            
            if filtro_tipo_validadas != "Todos":
                df_mostrar = df_mostrar[df_mostrar['tipo'] == filtro_tipo_validadas]
            
            if min_operaciones_validadas > 0:
                df_mostrar = df_mostrar[df_mostrar['operaciones'] >= min_operaciones_validadas]
            
            # Ordenar por factor de mejora (descendente)
            df_mostrar = df_mostrar.sort_values('factor_mejora', ascending=False)
            
            if len(df_mostrar) > 0:
                st.info(f"📋 Mostrando {len(df_mostrar)} combinaciones validadas")
                
                # Preparar DataFrame para mostrar
                display_df = df_mostrar.copy()
                display_df = display_df.round({
                    'rendimiento_total': 6,
                    'umbral_monkey': 6,
                    'margen_vs_monkey': 6,
                    'factor_mejora': 2,
                    'rendimiento_por_operacion': 6
                })
                
                # Reordenar columnas para la tabla final
                columnas_finales = [
                    'descripcion', 'tipo', 'regla_compuesta', 
                    'rendimiento_total', 'operaciones', 'umbral_monkey',
                    'margen_vs_monkey', 'factor_mejora'
                ]
                
                st.dataframe(
                    display_df[columnas_finales],
                    use_container_width=True,
                    height=500,
                    column_config={
                        'descripcion': st.column_config.TextColumn('Variables', width="medium"),
                        'tipo': st.column_config.TextColumn('Tipo', width="small"),
                        'regla_compuesta': st.column_config.TextColumn('Regla Compuesta', width="large"),
                        'rendimiento_total': st.column_config.NumberColumn('Rendimiento Total', format="%.6f"),
                        'operaciones': st.column_config.NumberColumn('Operaciones', format="%d"),
                        'umbral_monkey': st.column_config.NumberColumn('Umbral Monkey', format="%.6f"),
                        'margen_vs_monkey': st.column_config.NumberColumn('Margen vs Monkey', format="%.6f"),
                        'factor_mejora': st.column_config.NumberColumn('Factor Mejora', format="%.2fx")
                    }
                )
                
                # ================================================================================================
                # TOP 5 COMBINACIONES VALIDADAS
                # ================================================================================================
                
                st.markdown("### 🏆 Top 5 Mejores Combinaciones Validadas")
                
                top_5_validadas = df_mostrar.head(5)
                
                for i, (_, combo) in enumerate(top_5_validadas.iterrows(), 1):
                    with st.expander(f"🏆 #{i} - {combo['descripcion']} ({combo['tipo']}) - Factor: {combo['factor_mejora']:.2f}x"):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Métricas:**")
                            st.write(f"• Rendimiento Total: {combo['rendimiento_total']:.6f}")
                            st.write(f"• Operaciones: {combo['operaciones']}")
                            st.write(f"• Umbral Monkey: {combo['umbral_monkey']:.6f}")
                            st.write(f"• Margen vs Monkey: {combo['margen_vs_monkey']:.6f}")
                            st.write(f"• Factor Mejora: {combo['factor_mejora']:.2f}x")
                        
                        with col2:
                            st.markdown("**🎯 Regla Compuesta:**")
                            st.code(combo['regla_compuesta'])
                            st.markdown("**💡 Interpretación:**")
                            if combo['tipo'] == 'LONG':
                                st.write("📈 Cuando AMBAS condiciones se cumplen → **COMPRAR**")
                            else:
                                st.write("📉 Cuando AMBAS condiciones se cumplen → **VENDER**")
            
            else:
                st.warning("⚠️ No hay combinaciones validadas que cumplan los filtros aplicados.")
        
        else:
            st.warning("⚠️ Ninguna combinación superó el Monkey Test con la configuración actual.")
            st.info("💡 Considera ajustar el quantil o revisar la calidad de las reglas base.")
        
        # ================================================================================================
        # ANÁLISIS COMPARATIVO
        # ================================================================================================
        
        st.markdown("### 📊 Análisis Comparativo")
        
        analysis_tab1, analysis_tab2 = st.tabs([
            "📋 Todas vs Validadas",
            "📈 Gráficos Monkey Test"
        ])
        
        with analysis_tab1:
            st.markdown("#### 📊 Comparación: Todas las Combinaciones vs Solo Validadas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔢 Todas las Combinaciones:**")
                combo_stats_todas = analizar_combinaciones_reglas(df_combinaciones_todas)
                st.write(f"• Total: {combo_stats_todas['total_combinaciones']}")
                st.write(f"• Rendimiento promedio: {combo_stats_todas['rendimiento_promedio']:.6f}")
                st.write(f"• Rendimiento máximo: {combo_stats_todas['rendimiento_maximo']:.6f}")
                st.write(f"• Operaciones promedio: {combo_stats_todas['operaciones_promedio']:.1f}")
            
            with col2:
                st.markdown("**✅ Solo Validadas:**")
                if len(df_combinaciones_validadas) > 0:
                    combo_stats_validadas = analizar_combinaciones_reglas(df_combinaciones_validadas)
                    st.write(f"• Total: {combo_stats_validadas['total_combinaciones']}")
                    st.write(f"• Rendimiento promedio: {combo_stats_validadas['rendimiento_promedio']:.6f}")
                    st.write(f"• Rendimiento máximo: {combo_stats_validadas['rendimiento_maximo']:.6f}")
                    st.write(f"• Operaciones promedio: {combo_stats_validadas['operaciones_promedio']:.1f}")
                else:
                    st.write("• Total: 0")
                    st.write("• Sin datos disponibles")
        
        with analysis_tab2:
            st.markdown("#### 📈 Gráficos Monkey Test")
            
            # Tabs para LONG y SHORT
            monkey_tab_long, monkey_tab_short = st.tabs(["📈 Monkey Test LONG", "📉 Monkey Test SHORT"])
            
            with monkey_tab_long:
                fig_long = create_monkey_test_chart(stats_monkey_long, 
                                                  titulo_personalizado="Monkey Test LONG - Dataset Completo")
                st.plotly_chart(fig_long, use_container_width=True)
            
            with monkey_tab_short:
                fig_short = create_monkey_test_chart(stats_monkey_short,
                                                   titulo_personalizado="Monkey Test SHORT - Dataset Completo")
                st.plotly_chart(fig_short, use_container_width=True)
        
        # ================================================================================================
        # BOTÓN PARA RECONFIGURAR
        # ================================================================================================
        
        st.markdown("---")
        if st.button("🔄 Reconfigurar Validación", type="secondary"):
            st.session_state.step_validation = False
            st.session_state.step_ensemble = False
            for key in ['df_combinaciones_todas', 'df_combinaciones_validadas', 
                       'stats_monkey_long_combo', 'stats_monkey_short_combo',
                       'config_validacion_combo', 'df_completo_validacion']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

def ensemble_tab():
    """Tab ENSEMBLE - Análisis de Estabilidad Temporal"""
    st.header("🤖 ENSEMBLE - Análisis de Estabilidad Temporal")
    st.markdown("---")
    
    if not st.session_state.step_validation:
        st.warning("⚠️ Completa primero la validación de reglas")
        return
    
    # Verificar que hay reglas validadas
    if 'df_combinaciones_validadas' not in st.session_state or len(st.session_state.df_combinaciones_validadas) == 0:
        st.error("❌ No hay reglas validadas para analizar")
        st.info("💡 Vuelve al tab VALIDATION para generar reglas validadas.")
        return
    
    # Obtener datos necesarios
    df_completo = st.session_state.df_completo_validacion.copy()
    reglas_validadas = st.session_state.df_combinaciones_validadas.copy()
    
    # Información inicial
    long_validadas = reglas_validadas[reglas_validadas['tipo'] == 'LONG']
    short_validadas = reglas_validadas[reglas_validadas['tipo'] == 'SHORT']
    
    st.success(f"✅ Reglas validadas disponibles: {len(long_validadas)} LONG + {len(short_validadas)} SHORT = {len(reglas_validadas)} total")
    
    # ================================================================================================
    # CONFIGURACIÓN DE PARÁMETROS
    # ================================================================================================
    
    st.markdown("### ⚙️ Configuración del Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_trozos = st.slider(
            "📊 Número de trozos del histórico:",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            help="Divide el histórico en N trozos iguales para evaluar estabilidad"
        )
    
    with col2:
        n_reglas = st.slider(
            "🏆 Número de mejores reglas a mostrar:",
            min_value=1,
            max_value=min(50, len(reglas_validadas) // 2) if len(reglas_validadas) >= 2 else 1,
            value=min(10, len(reglas_validadas) // 2) if len(reglas_validadas) >= 2 else 1,
            step=1,
            help="Número de top reglas por tipo (LONG y SHORT)"
        )
    
    # Información del proceso
    dataset_size = len(df_completo.dropna(subset=['Target']))
    trozo_size = dataset_size // n_trozos
    
    st.info(
        f"🔄 **Proceso**: Dividir {dataset_size:,} observaciones en {n_trozos} trozos "
        f"(~{trozo_size:,} obs/trozo) → Evaluar {len(reglas_validadas)} reglas → "
        f"Seleccionar top {n_reglas} LONG + {n_reglas} SHORT por ratio de estabilidad"
    )
    
    # ================================================================================================
    # BOTÓN EJECUTAR ANÁLISIS
    # ================================================================================================
    
    if st.button("🚀 Ejecutar Análisis de Estabilidad", type="primary", use_container_width=True):
        
        try:
            with st.spinner(f'Analizando estabilidad en {n_trozos} trozos...'):
                
                # Calcular estabilidad de todas las reglas
                df_estabilidad = calcular_estabilidad_reglas(
                    df_completo,
                    reglas_validadas,
                    n_trozos=n_trozos
                )
                
                # Seleccionar mejores reglas
                top_long, top_short = seleccionar_mejores_reglas_por_estabilidad(
                    df_estabilidad,
                    n_reglas=n_reglas
                )
                
                # Guardar resultados en session state
                st.session_state.df_estabilidad_ensemble = df_estabilidad
                st.session_state.top_long_ensemble = top_long
                st.session_state.top_short_ensemble = top_short
                st.session_state.config_ensemble = {
                    'n_trozos': n_trozos,
                    'n_reglas': n_reglas
                }
                st.session_state.step_ensemble = True
                
                st.success(f"✅ Análisis completado: {len(top_long)} reglas LONG + {len(top_short)} reglas SHORT seleccionadas")
            
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            return
    
    # ================================================================================================
    # MOSTRAR RESULTADOS SI YA SE COMPLETÓ
    # ================================================================================================
    
    if st.session_state.get('step_ensemble', False) and 'top_long_ensemble' in st.session_state:
        
        top_long = st.session_state.top_long_ensemble
        top_short = st.session_state.top_short_ensemble
        config = st.session_state.config_ensemble
        
        st.markdown("---")
        st.markdown("### 📊 Resultados del Análisis de Estabilidad")
        
        # ================================================================================================
        # MÉTRICAS GENERALES
        # ================================================================================================
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Trozos Analizados", config['n_trozos'])
        
        with col2:
            st.metric("🏆 Reglas LONG", len(top_long))
        
        with col3:
            st.metric("🏆 Reglas SHORT", len(top_short))
        
        with col4:
            # Promedio de ratios
            if len(top_long) > 0 and len(top_short) > 0:
                ratio_promedio = (top_long['ratio_estabilidad'].mean() + top_short['ratio_estabilidad'].mean()) / 2
                st.metric("📈 Ratio Promedio", f"{ratio_promedio:.3f}")
            else:
                st.metric("📈 Ratio Promedio", "N/A")
        
        # ================================================================================================
        # TABLAS DE RESULTADOS
        # ================================================================================================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Top Reglas LONG por Estabilidad")
            
            if len(top_long) > 0:
                # Preparar DataFrame para mostrar
                display_long = top_long.copy()
                display_long = display_long.round({
                    'rendimiento_promedio': 6,
                    'std_rendimientos': 6,
                    'ratio_estabilidad': 3,
                    'rendimiento_total_original': 6
                })
                
                # Seleccionar columnas principales
                columnas_mostrar = [
                    'descripcion', 'rendimiento_promedio', 'std_rendimientos', 
                    'ratio_estabilidad', 'rendimiento_total_original'
                ]
                
                st.dataframe(
                    display_long[columnas_mostrar],
                    use_container_width=True,
                    height=400,
                    column_config={
                        'descripcion': st.column_config.TextColumn('Variables', width="medium"),
                        'rendimiento_promedio': st.column_config.NumberColumn('Rend. Promedio', format="%.6f"),
                        'std_rendimientos': st.column_config.NumberColumn('Desv. Std', format="%.6f"),
                        'ratio_estabilidad': st.column_config.NumberColumn('Ratio Estabilidad', format="%.3f"),
                        'rendimiento_total_original': st.column_config.NumberColumn('Rend. Original', format="%.6f")
                    }
                )
                
                # Mejor regla LONG
                if len(top_long) > 0:
                    mejor_long = top_long.iloc[0]
                    st.success(f"🥇 **Mejor LONG**: {mejor_long['descripcion']} (Ratio: {mejor_long['ratio_estabilidad']:.3f})")
            
            else:
                st.warning("⚠️ No hay reglas LONG suficientes para mostrar")
        
        with col2:
            st.markdown("### 📉 Top Reglas SHORT por Estabilidad")
            
            if len(top_short) > 0:
                # Preparar DataFrame para mostrar
                display_short = top_short.copy()
                display_short = display_short.round({
                    'rendimiento_promedio': 6,
                    'std_rendimientos': 6,
                    'ratio_estabilidad': 3,
                    'rendimiento_total_original': 6
                })
                
                # Seleccionar columnas principales
                columnas_mostrar = [
                    'descripcion', 'rendimiento_promedio', 'std_rendimientos', 
                    'ratio_estabilidad', 'rendimiento_total_original'
                ]
                
                st.dataframe(
                    display_short[columnas_mostrar],
                    use_container_width=True,
                    height=400,
                    column_config={
                        'descripcion': st.column_config.TextColumn('Variables', width="medium"),
                        'rendimiento_promedio': st.column_config.NumberColumn('Rend. Promedio', format="%.6f"),
                        'std_rendimientos': st.column_config.NumberColumn('Desv. Std', format="%.6f"),
                        'ratio_estabilidad': st.column_config.NumberColumn('Ratio Estabilidad', format="%.3f"),
                        'rendimiento_total_original': st.column_config.NumberColumn('Rend. Original', format="%.6f")
                    }
                )
                
                # Mejor regla SHORT
                if len(top_short) > 0:
                    mejor_short = top_short.iloc[0]
                    st.success(f"🥇 **Mejor SHORT**: {mejor_short['descripcion']} (Ratio: {mejor_short['ratio_estabilidad']:.3f})")
            
            else:
                st.warning("⚠️ No hay reglas SHORT suficientes para mostrar")
        
        # ================================================================================================
        # ANÁLISIS ADICIONAL
        # ================================================================================================
        
        st.markdown("### 🔍 Análisis Adicional")
        
        # Mostrar estadísticas comparativas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Estadísticas LONG")
            if len(top_long) > 0:
                st.write(f"• **Mejor ratio**: {top_long['ratio_estabilidad'].max():.3f}")
                st.write(f"• **Ratio promedio**: {top_long['ratio_estabilidad'].mean():.3f}")
                st.write(f"• **Rendimiento promedio**: {top_long['rendimiento_promedio'].mean():.6f}")
                st.write(f"• **Std promedio**: {top_long['std_rendimientos'].mean():.6f}")
            else:
                st.write("No hay datos disponibles")
        
        with col2:
            st.markdown("#### 📊 Estadísticas SHORT")
            if len(top_short) > 0:
                st.write(f"• **Mejor ratio**: {top_short['ratio_estabilidad'].max():.3f}")
                st.write(f"• **Ratio promedio**: {top_short['ratio_estabilidad'].mean():.3f}")
                st.write(f"• **Rendimiento promedio**: {top_short['rendimiento_promedio'].mean():.6f}")
                st.write(f"• **Std promedio**: {top_short['std_rendimientos'].mean():.6f}")
            else:
                st.write("No hay datos disponibles")
        
        # ================================================================================================
        # INTERPRETACIÓN
        # ================================================================================================
        
        st.markdown("### 💡 Interpretación de Resultados")
        
        st.info("""
        **Ratio de Estabilidad = Rendimiento Promedio / Desviación Estándar**
        
        • **Ratio alto**: Regla con rendimientos consistentes y predecibles
        • **Ratio bajo**: Regla con rendimientos variables e impredecibles
        • **Las mejores reglas** combinan buen rendimiento promedio con baja variabilidad
        """)
        
        # ================================================================================================
        # BOTÓN PARA RECONFIGURAR
        # ================================================================================================
        
        st.markdown("---")
        if st.button("🔄 Reconfigurar Análisis", type="secondary"):
            # Reset del ensemble
            st.session_state.step_ensemble = False
            for key in ['df_estabilidad_ensemble', 'top_long_ensemble', 'top_short_ensemble', 'config_ensemble']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

def visualization_tab():
    """Tab VISUALIZATION - Visualizaciones avanzadas del pipeline"""
    st.header("📊 VISUALIZATION - Análisis Visual Avanzado")
    st.markdown("---")
    
    if not st.session_state.get('step_ensemble', False):
        st.warning("⚠️ Completa primero el análisis ENSEMBLE para acceder a las visualizaciones")
        return
    
    # Obtener datos necesarios
    df_completo = st.session_state.df_completo_validacion.copy()
    top_long = st.session_state.top_long_ensemble.copy()
    top_short = st.session_state.top_short_ensemble.copy()
    
    # DEBUG: Verificar estructura
    debug_ensemble_structure(top_long, top_short)
    
    st.success(f"✅ Datos disponibles: {len(top_long)} reglas LONG + {len(top_short)} reglas SHORT del ENSEMBLE")
    
    # ================================================================================================
    # EQUITY CURVES PRINCIPAL
    # ================================================================================================
    
    st.markdown("### 📈 Curvas de Equity")
    
    if len(top_long) == 0 and len(top_short) == 0:
        st.warning("⚠️ No hay reglas del ENSEMBLE para mostrar equity curves")
        return
    
    # Selector de reglas para equity curve
    col1, col2 = st.columns(2)
    
    with col1:
        max_reglas = len(top_long) + len(top_short)
        n_reglas_equity = st.slider(
            "📊 Número de reglas a mostrar:",
            min_value=1,
            max_value=min(20, max_reglas),
            value=min(10, max_reglas),
            help="Top reglas por ratio de estabilidad del ENSEMBLE"
        )
    
    with col2:
        show_combined = st.checkbox(
            "📊 Mostrar portfolio combinado",
            value=True,
            help="Combinar todas las reglas seleccionadas en un solo portfolio"
        )
    
    st.info(f"🎯 Se mostrarán las {n_reglas_equity} mejores reglas por ratio de estabilidad")
    
    if st.button("🚀 Generar Equity Curves", type="primary", use_container_width=True):
        with st.spinner('Calculando curvas de equity...'):
            
            try:
                # Seleccionar reglas del ENSEMBLE (CORREGIDO)
                top_reglas_combined = []
                
                print(f"🔍 DEBUG - Seleccionando reglas:")
                print(f"   top_long disponibles: {len(top_long)}")
                print(f"   top_short disponibles: {len(top_short)}")
                print(f"   n_reglas_equity solicitadas: {n_reglas_equity}")
                
                # Añadir reglas LONG
                if len(top_long) > 0:
                    n_long = min(n_reglas_equity // 2, len(top_long))
                    print(f"   Seleccionando {n_long} reglas LONG")
                    
                    reglas_long_selected = top_long.head(n_long)
                    for idx, regla in reglas_long_selected.iterrows():
                        regla_dict = {
                            'descripcion': regla['descripcion'],
                            'tipo': regla['tipo'],
                            'regla_compuesta': regla['regla_compuesta'],
                            'ratio_estabilidad': regla.get('ratio_estabilidad', 0),
                            'rendimiento_total_original': regla.get('rendimiento_total_original', 0),
                            'operaciones_total_original': regla.get('operaciones_total_original', 0)
                        }
                        top_reglas_combined.append(regla_dict)
                        print(f"      ✅ LONG: {regla['descripcion']}")
                
                # Añadir reglas SHORT
                if len(top_short) > 0:
                    n_short = min(n_reglas_equity - len(top_reglas_combined), len(top_short))
                    print(f"   Seleccionando {n_short} reglas SHORT")
                    
                    reglas_short_selected = top_short.head(n_short)
                    for idx, regla in reglas_short_selected.iterrows():
                        regla_dict = {
                            'descripcion': regla['descripcion'],
                            'tipo': regla['tipo'],
                            'regla_compuesta': regla['regla_compuesta'],
                            'ratio_estabilidad': regla.get('ratio_estabilidad', 0),
                            'rendimiento_total_original': regla.get('rendimiento_total_original', 0),
                            'operaciones_total_original': regla.get('operaciones_total_original', 0)
                        }
                        top_reglas_combined.append(regla_dict)
                        print(f"      ✅ SHORT: {regla['descripcion']}")
                
                print(f"   📊 Total reglas seleccionadas: {len(top_reglas_combined)}")
                
                if len(top_reglas_combined) == 0:
                    st.error("❌ No se pudieron seleccionar reglas válidas")
                    return
                
                st.info(f"📊 Procesando {len(top_reglas_combined)} reglas: {len([r for r in top_reglas_combined if r['tipo']=='LONG'])} LONG + {len([r for r in top_reglas_combined if r['tipo']=='SHORT'])} SHORT")
                
                # Generar equity curves
                equity_data = generar_equity_curves(df_completo, top_reglas_combined)
                
                if len(equity_data) == 0:
                    st.error("❌ No se pudieron generar equity curves")
                    return
                
                st.success(f"✅ Equity curves generadas para {len(equity_data)} reglas")
                
                # Gráfico individual de cada regla (SIN LEYENDA DETALLADA)
                st.markdown("#### 📈 Equity Curves Individuales")
                fig_individual = crear_grafico_equity_individual_simple(equity_data)
                st.plotly_chart(fig_individual, use_container_width=True)
                
                if show_combined:
                    # Gráfico del portfolio combinado
                    st.markdown("#### 🏆 Portfolio Combinado vs Benchmark")
                    fig_combined = crear_grafico_equity_combinado(equity_data)
                    st.plotly_chart(fig_combined, use_container_width=True)
                
                # ================================================================================================
                # TABLAS SEPARADAS POR TIPO (COMO EN ENSEMBLE)
                # ================================================================================================
                
                st.markdown("### 📊 Reglas Utilizadas")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Reglas LONG")
                    
                    reglas_long_mostradas = [r for r in top_reglas_combined if r['tipo'] == 'LONG']
                    
                    if len(reglas_long_mostradas) > 0:
                        # Preparar DataFrame para LONG
                        display_long = pd.DataFrame(reglas_long_mostradas)
                        display_long_table = display_long[['regla_compuesta', 'rendimiento_total_original', 'ratio_estabilidad', 'operaciones_total_original']].copy()
                        display_long_table = display_long_table.rename(columns={
                            'regla_compuesta': 'Regla',
                            'rendimiento_total_original': 'Rendimiento',
                            'ratio_estabilidad': 'Ratio Estabilidad',
                            'operaciones_total_original': 'Operaciones'
                        })
                        display_long_table = display_long_table.round({'Rendimiento': 6, 'Ratio Estabilidad': 3})
                        
                        st.dataframe(
                            display_long_table,
                            use_container_width=True,
                            height=min(400, len(display_long_table) * 40 + 100)
                        )
                    else:
                        st.info("No hay reglas LONG en esta selección")
                
                with col2:
                    st.markdown("### 📉 Reglas SHORT")
                    
                    reglas_short_mostradas = [r for r in top_reglas_combined if r['tipo'] == 'SHORT']
                    
                    if len(reglas_short_mostradas) > 0:
                        # Preparar DataFrame para SHORT
                        display_short = pd.DataFrame(reglas_short_mostradas)
                        display_short_table = display_short[['regla_compuesta', 'rendimiento_total_original', 'ratio_estabilidad', 'operaciones_total_original']].copy()
                        display_short_table = display_short_table.rename(columns={
                            'regla_compuesta': 'Regla',
                            'rendimiento_total_original': 'Rendimiento',
                            'ratio_estabilidad': 'Ratio Estabilidad',
                            'operaciones_total_original': 'Operaciones'
                        })
                        display_short_table = display_short_table.round({'Rendimiento': 6, 'Ratio Estabilidad': 3})
                        
                        st.dataframe(
                            display_short_table,
                            use_container_width=True,
                            height=min(400, len(display_short_table) * 40 + 100)
                        )
                    else:
                        st.info("No hay reglas SHORT en esta selección")
                
            except Exception as e:
                st.error(f"❌ Error generando equity curves: {str(e)}")
                st.error("💡 Verifica que las reglas del ENSEMBLE sean válidas")

def optimization_tab():
    """Tab OPTIMIZATION - Análisis de optimización MT5"""
    st.header("⚙️ OPTIMIZATION - Análisis de Optimización MT5")
    st.markdown("---")
    
    # Información del proceso
    st.info(
        "📋 **Proceso**: Sube el reporte XML de optimización de MT5 → "
        "Análisis estadístico → Rangos recomendados para parámetros"
    )
    
    # Mostrar estado del pipeline anterior
    if not st.session_state.get('step_ensemble', False):
        st.warning("⚠️ Recomendado: Completa primero el análisis ENSEMBLE para obtener reglas optimizadas")
    else:
        st.success("✅ Pipeline completado - Listo para analizar optimización de parámetros")
    
    # ================================================================================================
    # UPLOADER DE ARCHIVO XML
    # ================================================================================================
    
    st.markdown("### 📁 Cargar Reporte de Optimización")
    
    uploaded_xml = st.file_uploader(
        "Arrastra y suelta tu archivo XML de optimización MT5",
        type=['xml'],
        help="Archivo generado por MetaTrader 5 después de ejecutar una optimización"
    )
    
    if uploaded_xml is None:
        st.info("👆 Carga un archivo XML para comenzar el análisis")
        
        # Mostrar instrucciones
        with st.expander("📖 ¿Cómo generar el archivo XML en MT5?", expanded=False):
            st.markdown("""
            **Pasos para obtener el reporte de optimización:**
            
            1. **Optimizar EA en MT5**: Strategy Tester → Optimization
            2. **Finalizar optimización**: Esperar a que termine el proceso
            3. **Exportar resultados**: Click derecho en tabla de resultados → "Save as Report"
            4. **Seleccionar formato**: Elegir "XML files (*.xml)"
            5. **Guardar archivo**: Dar nombre y guardar
            6. **Subir aquí**: Usar el uploader de arriba
            
            **Requisitos del archivo:**
            - Debe contener columna "Result" con los valores de fitness
            - Debe incluir columnas de parámetros optimizados
            - Formato XML estándar de MT5
            """)
        return
    
    # ================================================================================================
    # PROCESAR ARCHIVO XML
    # ================================================================================================
    
    if uploaded_xml is not None:
        
        with st.spinner('📊 Analizando archivo XML...'):
            
            # Analizar archivo
            analisis_xml = analizar_archivo_optimizacion_mt5(uploaded_xml)
            
            if not analisis_xml['success']:
                st.error(f"❌ Error procesando archivo: {analisis_xml['error']}")
                return
            
            # Obtener DataFrame
            df_optimizacion = analisis_xml['dataframe']
            
            st.success(f"✅ Archivo procesado: {analisis_xml['total_filas']:,} filas, {len(analisis_xml['columnas'])} columnas")
            
            # Mostrar información básica
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Pruebas", f"{len(df_optimizacion):,}")
            
            with col2:
                if 'Result' in df_optimizacion.columns:
                    resultados_positivos = len(df_optimizacion[df_optimizacion['Result'] > 0])
                    st.metric("✅ Resultados > 0", f"{resultados_positivos:,}")
                else:
                    st.metric("❌ Sin columna 'Result'", "0")
            
            with col3:
                st.metric("📋 Columnas Detectadas", len(analisis_xml['columnas']))
            
            # Mostrar muestra de datos
            st.markdown("### 👀 Vista Previa de Datos")
            st.dataframe(df_optimizacion.head(10), use_container_width=True)
            
            # ================================================================================================
            # CALCULAR RANGOS DE PARÁMETROS
            # ================================================================================================
            
            st.markdown("---")
            
            if st.button("🚀 Calcular Rangos de Parámetros", type="primary", use_container_width=True):
                
                with st.spinner('🧮 Calculando rangos recomendados...'):
                    
                    # Calcular rangos
                    analisis_rangos = calcular_rangos_parametros_optimizacion(df_optimizacion)
                    
                    if not analisis_rangos['success']:
                        st.error(f"❌ Error calculando rangos: {analisis_rangos['error']}")
                        return
                    
                    # Guardar en session state
                    st.session_state.analisis_optimizacion = analisis_rangos
                    st.session_state.step_optimization = True
                    
                    st.success("✅ Rangos calculados correctamente")
            
            # ================================================================================================
            # MOSTRAR RESULTADOS SI YA SE CALCULARON
            # ================================================================================================
            
            if st.session_state.get('step_optimization', False) and 'analisis_optimizacion' in st.session_state:
                
                analisis = st.session_state.analisis_optimizacion
                
                st.markdown("---")
                st.markdown("### 📊 Resultados del Análisis")
                
                # ================================================================================================
                # MÉTRICAS DEL FILTRADO
                # ================================================================================================
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Pruebas Totales", f"{analisis['total_filas_original']:,}")
                
                with col2:
                    st.metric("✅ Result > 0", f"{analisis['filas_result_positivo']:,}")
                
                with col3:
                    st.metric("📈 Result ≥ Promedio", f"{analisis['filas_finales']:,}")
                
                with col4:
                    tasa_exito = analisis['filas_finales'] / analisis['total_filas_original'] * 100
                    st.metric("🎯 Tasa de Éxito", f"{tasa_exito:.1f}%")
                
                # ================================================================================================
                # INFORMACIÓN DEL PROCESO
                # ================================================================================================
                
                st.markdown("### 🔍 Proceso de Filtrado")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📋 Estadísticas del Filtrado:**")
                    st.write(f"• **Pruebas originales**: {analisis['total_filas_original']:,}")
                    st.write(f"• **Con Result > 0**: {analisis['filas_result_positivo']:,}")
                    st.write(f"• **Promedio de Result**: {analisis['result_promedio']}")
                    st.write(f"• **Finales (≥ promedio)**: {analisis['filas_finales']:,}")
                
                with col2:
                    st.markdown("**🗑️ Columnas Eliminadas:**")
                    if analisis['columnas_eliminadas']:
                        for col in analisis['columnas_eliminadas'][:5]:  # Mostrar máximo 5
                            st.write(f"• {col}")
                        if len(analisis['columnas_eliminadas']) > 5:
                            st.write(f"• ... y {len(analisis['columnas_eliminadas']) - 5} más")
                    else:
                        st.write("• Ninguna columna eliminada")
                
                # ================================================================================================
                # RANGOS RECOMENDADOS - TABLA PRINCIPAL
                # ================================================================================================
                
                st.markdown("### 🎯 Rangos Recomendados para Parámetros")
                
                if analisis['rangos_parametros']:
                    
                    # Preparar datos para la tabla
                    tabla_rangos = []
                    
                    for parametro, datos in analisis['rangos_parametros'].items():
                        tabla_rangos.append({
                            'Parámetro': parametro,
                            'Tipo': datos['tipo'].title(),
                            'Rango Mínimo': datos['min'],
                            'Rango Máximo': datos['max'],
                            'Media': datos['media'],
                            'Desv. Estándar': datos['std'],
                            'Rango Completo': f"{datos['min']} - {datos['max']}"
                        })
                    
                    df_tabla_rangos = pd.DataFrame(tabla_rangos)
                    
                    # Mostrar tabla
                    st.dataframe(
                        df_tabla_rangos,
                        use_container_width=True,
                        height=min(500, len(df_tabla_rangos) * 40 + 100),
                        column_config={
                            'Parámetro': st.column_config.TextColumn('Parámetro', width="medium"),
                            'Tipo': st.column_config.TextColumn('Tipo', width="small"),
                            'Rango Mínimo': st.column_config.NumberColumn('Mín'),
                            'Rango Máximo': st.column_config.NumberColumn('Máx'),
                            'Media': st.column_config.NumberColumn('Media', format="%.4f"),
                            'Desv. Estándar': st.column_config.NumberColumn('Std', format="%.4f"),
                            'Rango Completo': st.column_config.TextColumn('Rango Completo', width="medium")
                        }
                    )
                    
                    # ================================================================================================
                    # RESUMEN EJECUTIVO
                    # ================================================================================================
                    
                    st.markdown("### 📋 Resumen Ejecutivo")
                    
                    st.success(
                        f"🎯 **{len(analisis['rangos_parametros'])} parámetros analizados** "
                        f"basados en {analisis['filas_finales']:,} pruebas exitosas "
                        f"({tasa_exito:.1f}% del total)"
                    )
                    
                    # Mostrar rangos en formato limpio
                    with st.expander("📊 Rangos Resumidos para Copy/Paste", expanded=False):
                        st.markdown("**Rangos calculados (Media ± 1 Desviación Estándar):**")
                        
                        rangos_texto = []
                        for parametro, datos in analisis['rangos_parametros'].items():
                            if datos['tipo'] == 'entero':
                                rangos_texto.append(f"• **{parametro}**: {datos['min']} - {datos['max']}")
                            else:
                                rangos_texto.append(f"• **{parametro}**: {datos['min']} - {datos['max']}")
                        
                        for linea in rangos_texto:
                            st.markdown(linea)
                    
                    # ================================================================================================
                    # GRÁFICOS DE DISTRIBUCIÓN
                    # ================================================================================================
                    
                    st.markdown("### 📊 Distribución de Parámetros")
                    
                    # Generar gráfico
                    fig_distribucion = crear_grafico_distribucion_parametros(analisis)
                    
                    if fig_distribucion:
                        st.plotly_chart(fig_distribucion, use_container_width=True)
                    else:
                        st.warning("⚠️ No se pudo generar el gráfico de distribución")
                    
                    # ================================================================================================
                    # TOP 5 MEJORES RESULTADOS
                    # ================================================================================================
                    
                    st.markdown("### 🏆 Top 5 Mejores Resultados")
                    
                    if 'mejores_resultados' in analisis and len(analisis['mejores_resultados']) > 0:
                        
                        mejores_df = analisis['mejores_resultados'].copy()
                        
                        # Redondear valores para mejor visualización
                        for col in mejores_df.select_dtypes(include=[np.number]).columns:
                            if col != 'Result':
                                mejores_df[col] = mejores_df[col].round(4)
                        
                        st.dataframe(mejores_df, use_container_width=True)
                        
                        # Mostrar el mejor resultado
                        mejor_resultado = mejores_df.iloc[0]
                        st.success(f"🥇 **Mejor resultado**: {mejor_resultado['Result']}")
                    
                    # ================================================================================================
                    # DESCARGAS
                    # ================================================================================================
                    
                    st.markdown("### 💾 Descargar Resultados")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Descarga de rangos CSV
                        csv_rangos = df_tabla_rangos.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📊 Descargar Rangos (CSV)",
                            data=csv_rangos,
                            file_name=f"rangos_parametros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Descarga de resumen completo
                        resumen_completo = generar_resumen_optimizacion(analisis)
                        st.download_button(
                            label="📋 Descargar Resumen (TXT)",
                            data=resumen_completo,
                            file_name=f"resumen_optimizacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
                else:
                    st.warning("⚠️ No se encontraron parámetros para analizar")
                
                # ================================================================================================
                # BOTÓN PARA RESET
                # ================================================================================================
                
                st.markdown("---")
                if st.button("🔄 Analizar Nuevo Archivo", type="secondary"):
                    st.session_state.step_optimization = False
                    if 'analisis_optimizacion' in st.session_state:
                        del st.session_state.analisis_optimizacion
                    st.rerun()

def sizing_tab():
    """Tab SIZING - Análisis de Position Sizing con Monte Carlo"""
    st.header("💰 SIZING - Análisis de Position Sizing")
    st.markdown("---")
    
    # Información del proceso
    st.info(
        "📋 **Proceso**: Sube el reporte HTML de backtest MT5 → "
        "Análisis Monte Carlo → Estimación de drawdown máximo probable para dimensionar posiciones"
    )
    
    # ================================================================================================
    # UPLOADER DE ARCHIVO HTML
    # ================================================================================================
    
    st.markdown("### 📁 Cargar Reporte de Backtest MT5")
    
    uploaded_html = st.file_uploader(
        "Arrastra y suelta tu archivo HTML de backtest MT5",
        type=['html', 'htm'],
        help="Archivo HTML generado por MetaTrader 5 después de ejecutar un backtest"
    )
    
    if uploaded_html is None:
        st.info("👆 Carga un archivo HTML para comenzar el análisis")
        
        # Mostrar instrucciones
        with st.expander("📖 ¿Cómo generar el archivo HTML en MT5?", expanded=False):
            st.markdown("""
            **Pasos para obtener el reporte de backtest:**
            
            1. **Ejecutar backtest en MT5**: Strategy Tester → Start
            2. **Finalizar backtest**: Esperar a que termine el proceso
            3. **Generar reporte**: Click derecho en la gráfica de resultados → "Save as Report"
            4. **Seleccionar formato**: Elegir "Web page, HTML only (*.html)"
            5. **Guardar archivo**: Dar nombre y guardar
            6. **Subir aquí**: Usar el uploader de arriba
            
            **Requisitos del archivo:**
            - Debe contener tabla de operaciones (deals)
            - Debe incluir fechas de inicio y fin del backtest
            - Formato HTML estándar de MT5
            """)
        return
    
    # ================================================================================================
    # PROCESAR ARCHIVO HTML
    # ================================================================================================
    
    if uploaded_html is not None:
        
        with st.spinner('📊 Analizando reporte de backtest...'):
            
            # Analizar archivo
            analisis_mt5 = analizar_reporte_mt5_html(uploaded_html)
            
            if not analisis_mt5['success']:
                st.error(f"❌ Error procesando archivo: {analisis_mt5['error']}")
                return
            
            # Guardar en session state
            st.session_state.analisis_mt5_sizing = analisis_mt5
            
            st.success(f"✅ Archivo procesado: {analisis_mt5['num_operaciones']:,} operaciones extraídas")
            
            # Mostrar información básica del backtest
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Operaciones", f"{analisis_mt5['num_operaciones']:,}")
            
            with col2:
                if analisis_mt5['fecha_inicio'] and analisis_mt5['fecha_fin']:
                    duracion = (analisis_mt5['fecha_fin'] - analisis_mt5['fecha_inicio']).days
                    st.metric("📅 Duración (días)", f"{duracion:,}")
                else:
                    st.metric("📅 Duración", "No detectada")
            
            with col3:
                st.metric("💰 Retorno Total", f"{analisis_mt5['suma_total']:.2f}")
            
            # ================================================================================================
            # CÁLCULO AUTOMÁTICO DE OPERACIONES POR AÑO
            # ================================================================================================
            
            st.markdown("---")
            st.markdown("### 📈 Cálculo de Operaciones por Año")

            # Intentar calcular automáticamente
            if analisis_mt5['fecha_inicio'] and analisis_mt5['fecha_fin']:
                calculo_ops = calcular_operaciones_por_año(
                    analisis_mt5['fecha_inicio'],
                    analisis_mt5['fecha_fin'],
                    analisis_mt5['num_operaciones']
                )                

                if calculo_ops['success']:                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📅 Duración del Backtest", f"{calculo_ops['duracion_años']} años")
                    
                    with col2:                        
                        st.metric("🔢 Operaciones/Año (Calculado)", f"{calculo_ops['operaciones_por_año']:.1f}")

                    with col3:
                        st.metric("🎯 Ops/Año (Entero)", f"{calculo_ops['operaciones_por_año_entero']}")
                    
                    # Permitir ajuste manual
                    st.markdown("**Ajustar operaciones por año (opcional):**")
                    operaciones_por_año = st.number_input(
                        "Operaciones por año para simulación:",
                        min_value=1,
                        # max_value=max(100000,analisis_mt5['num_operaciones']),
                        max_value=analisis_mt5['num_operaciones'],
                        value=calculo_ops['operaciones_por_año_entero'],
                        help="Número de operaciones que se ejecutarían en un año típico"
                    )
                    
                else:
                    st.warning(f"⚠️ Error calculando operaciones por año: {calculo_ops['error']}")
                    operaciones_por_año = st.number_input(
                        "Operaciones por año (manual):",
                        min_value=1,
                        max_value=analisis_mt5['num_operaciones'],
                        value=min(100, analisis_mt5['num_operaciones']),
                        help="Estima cuántas operaciones ejecutarías en un año"
                    )
            else:
                st.warning("⚠️ No se pudieron detectar fechas automáticamente")
                operaciones_por_año = st.number_input(
                    "Operaciones por año (manual):",
                    min_value=1,
                    max_value=analisis_mt5['num_operaciones'],
                    value=min(100, analisis_mt5['num_operaciones']),
                    help="Estima cuántas operaciones ejecutarías en un año"
                )
            
            # ================================================================================================
            # CONFIGURACIÓN DE SIMULACIÓN MONTE CARLO
            # ================================================================================================
            
            st.markdown("---")
            st.markdown("### ⚙️ Configuración Monte Carlo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_simulaciones = st.selectbox(
                    "🔄 Número de Simulaciones:",
                    [1000, 2000, 5000, 10000, 20000],
                    index=3,  # 10000 por defecto
                    help="Más simulaciones = mayor precisión pero más tiempo"
                )
            
            with col2:
                capital_inicial = st.number_input(
                    "💰 Capital Inicial:",
                    min_value=100,
                    max_value=1000000,
                    value=2000,
                    step=100,
                    help="Capital inicial para cada simulación"
                )
            
            with col3:
                # Calcular tiempo estimado
                tiempo_estimado = max(1, n_simulaciones // 2000)
                st.metric("⏱️ Tiempo Estimado", f"~{tiempo_estimado} min")
            
            # Resumen de configuración
            st.info(
                f"🎯 **Configuración**: {n_simulaciones:,} simulaciones de {operaciones_por_año} operaciones "
                f"cada una, capital inicial: {capital_inicial:,}"
            )
            
            # ================================================================================================
            # BOTÓN EJECUTAR SIMULACIÓN
            # ================================================================================================
            
            if st.button("🚀 Ejecutar Simulación Monte Carlo", type="primary", use_container_width=True):
                
                try:
                    with st.spinner('🎲 Ejecutando simulaciones Monte Carlo...'):
                        
                        # Ejecutar simulación
                        simulacion_resultado = simulacion_monte_carlo_sizing(
                            analisis_mt5['retornos'],
                            operaciones_por_año,
                            n_simulaciones,
                            capital_inicial
                        )
                        
                        if not simulacion_resultado['success']:
                            st.error(f"❌ Error en simulación: {simulacion_resultado['error']}")
                            return
                        
                        # Guardar resultados en session state
                        st.session_state.simulacion_sizing = simulacion_resultado
                        st.session_state.step_sizing = True
                        
                        st.success(f"✅ Simulación completada: {n_simulaciones:,} iteraciones ejecutadas")
                        
                except Exception as e:
                    st.error(f"❌ Error ejecutando simulación: {str(e)}")
                    return
            
            # ================================================================================================
            # MOSTRAR RESULTADOS SI YA SE COMPLETÓ
            # ================================================================================================
            
            if st.session_state.get('step_sizing', False) and 'simulacion_sizing' in st.session_state:
                
                simulacion_resultado = st.session_state.simulacion_sizing
                
                st.markdown("---")
                st.markdown("### 📊 Resultados de la Simulación")
                
                # ================================================================================================
                # MÉTRICAS PRINCIPALES
                # ================================================================================================
                
                st.markdown("#### 💰 Métricas de Drawdown Máximo")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Media", f"{simulacion_resultado['dd_mean']:.2f}")
                
                with col2:
                    st.metric("📈 Mediana", f"{simulacion_resultado['dd_median']:.2f}")
                
                with col3:
                    st.metric("📉 Percentil 95%", f"{simulacion_resultado['dd_percentiles'][95]:.2f}")
                
                with col4:
                    st.metric("🔴 Percentil 99%", f"{simulacion_resultado['dd_percentiles'][99]:.2f}")
                
                # ================================================================================================
                # GRÁFICOS PRINCIPALES
                # ================================================================================================
                
                st.markdown("#### 📈 Gráficos de Distribución")
                
                # Tabs para diferentes gráficos
                graph_tab1, graph_tab2, graph_tab3 = st.tabs([
                    "📉 Distribución Drawdown",
                    "💰 Distribución Profit", 
                    "📊 Drawdown Histórico"
                ])
                
                with graph_tab1:
                    fig_dd = crear_grafico_distribucion_monte_carlo(simulacion_resultado, 'drawdown')
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                with graph_tab2:
                    fig_profit = crear_grafico_distribucion_monte_carlo(simulacion_resultado, 'profit')
                    st.plotly_chart(fig_profit, use_container_width=True)
                
                with graph_tab3:
                    # Calcular drawdown del histórico completo
                    dd_historico = calcular_maxdd(analisis_mt5['retornos'], capital_inicial)
                    fig_historico = crear_grafico_drawdown_historico(dd_historico)
                    st.plotly_chart(fig_historico, use_container_width=True)
                
                # ================================================================================================
                # TABLA DE PERCENTILES
                # ================================================================================================
                
                st.markdown("#### 📋 Tabla de Percentiles")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📉 Drawdown Máximo:**")
                    percentiles_dd_data = []
                    for p, val in simulacion_resultado['dd_percentiles'].items():
                        percentiles_dd_data.append({
                            'Percentil': f'{p}%',
                            'Drawdown': f'{val:.2f}',
                            'Probabilidad': f'{100-p}%'
                        })
                    
                    df_percentiles_dd = pd.DataFrame(percentiles_dd_data)
                    st.dataframe(df_percentiles_dd, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**💰 Profit Anual:**")
                    percentiles_profit_data = []
                    for p, val in simulacion_resultado['profit_percentiles'].items():
                        percentiles_profit_data.append({
                            'Percentil': f'{p}%',
                            'Profit': f'{val:.2f}',
                            'Probabilidad': f'{p}%'
                        })
                    
                    df_percentiles_profit = pd.DataFrame(percentiles_profit_data)
                    st.dataframe(df_percentiles_profit, use_container_width=True, hide_index=True)
                
                # ================================================================================================
                # RECOMENDACIONES DE POSITION SIZING
                # ================================================================================================
                
                st.markdown("#### 🎯 Recomendaciones de Position Sizing")
                
                p95_dd = simulacion_resultado['dd_percentiles'][95]
                p99_dd = simulacion_resultado['dd_percentiles'][99]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Para 95% de confianza:**")
                    st.success(f"💰 Preparar capital para DD máximo: **{p95_dd:.2f}**")
                    if capital_inicial > 0 and p95_dd > 0:
                        ratio_95 = capital_inicial / p95_dd
                        st.info(f"📈 Ratio de seguridad: **{ratio_95:.2f}x**")
                
                with col2:
                    st.markdown("**📊 Para 99% de confianza:**")
                    st.error(f"💰 Preparar capital para DD máximo: **{p99_dd:.2f}**")
                    if capital_inicial > 0 and p99_dd > 0:
                        ratio_99 = capital_inicial / p99_dd
                        st.info(f"📈 Ratio de seguridad: **{ratio_99:.2f}x**")
                
                # Interpretación adicional
                st.markdown("**💡 Interpretación:**")
                st.write(f"• En el **95%** de los casos, el drawdown máximo será ≤ {p95_dd:.2f}")
                st.write(f"• En el **99%** de los casos, el drawdown máximo será ≤ {p99_dd:.2f}")
                st.write(f"• En el **5%** de los casos, el drawdown podría superar {p95_dd:.2f}")
                st.write(f"• En el **1%** de los casos, el drawdown podría superar {p99_dd:.2f}")
                
                # ================================================================================================
                # COMPARACIÓN CON HISTÓRICO
                # ================================================================================================
                
                st.markdown("#### 🔍 Comparación con Histórico")
                
                dd_historico = calcular_maxdd(analisis_mt5['retornos'], capital_inicial)
                dd_hist_max = dd_historico['max_dd_abs']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 DD Histórico Real", f"{dd_hist_max:.2f}")
                
                with col2:
                    percentil_historico = np.percentile(simulacion_resultado['max_dd_list'], 
                                                      [p for p, v in simulacion_resultado['dd_percentiles'].items() 
                                                       if v >= dd_hist_max])
                    percentil_hist = min(percentil_historico) if len(percentil_historico) > 0 else 100
                    st.metric("📈 Percentil del Histórico", f"~{percentil_hist:.0f}%")
                
                with col3:
                    comparacion = "Normal" if dd_hist_max <= p95_dd else ("Alto" if dd_hist_max <= p99_dd else "Extremo")
                    color = "🟢" if comparacion == "Normal" else ("🟡" if comparacion == "Alto" else "🔴")
                    st.metric("🎯 Nivel Histórico", f"{color} {comparacion}")
                
                # ================================================================================================
                # DESCARGA DE RESULTADOS
                # ================================================================================================
                
                st.markdown("---")
                st.markdown("### 💾 Descargar Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generar reporte completo
                    reporte_completo = generar_reporte_sizing(analisis_mt5, simulacion_resultado)
                    
                    st.download_button(
                        label="📋 Descargar Reporte Completo",
                        data=reporte_completo,
                        file_name=f"reporte_sizing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Generar CSV con resultados detallados
                    df_resultados = pd.DataFrame({
                        'Simulacion': range(1, len(simulacion_resultado['max_dd_list']) + 1),
                        'Drawdown_Maximo': simulacion_resultado['max_dd_list'],
                        'Profit_Total': simulacion_resultado['profits_list']
                    })
                    
                    csv_resultados = df_resultados.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Descargar Datos CSV",
                        data=csv_resultados,
                        file_name=f"simulaciones_sizing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # ================================================================================================
                # BOTÓN PARA NUEVA SIMULACIÓN
                # ================================================================================================
                
                st.markdown("---")
                if st.button("🔄 Nueva Simulación", type="secondary"):
                    st.session_state.step_sizing = False
                    if 'simulacion_sizing' in st.session_state:
                        del st.session_state.simulacion_sizing
                    st.rerun()

# ================================================================================================
# FUNCIÓN PRINCIPAL
# ================================================================================================

def main():
    """Aplicación principal"""
    initialize_session_state()
    
    # Título principal
    st.title("📈 Trading Analytics App")
    
    # ESTRUCTURA DE TABS MODIFICADA - AÑADIR EL NUEVO TAB SIZING
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 LOAD", 
        "🎯 TARGET", 
        "🔄 SPLIT", 
        "🎯 FEATURE SELECTION", 
        "📦 EXTRACTION", 
        "✔️ VALIDATION", 
        "🤖 ENSEMBLE",
        "📊 VISUALIZATION",
        "⚙️ OPTIMIZATION",
        "💰 SIZING"          # ← NUEVO TAB
    ])
    
    with tab1:
        load_tab()
    
    with tab2:
        target_tab()
    
    with tab3:
        split_tab()
    
    with tab4:
        feature_selection_tab()
    
    with tab5:
        extraction_tab()
    
    with tab6:
        validation_tab()
    
    with tab7:
        ensemble_tab()
    
    with tab8:
        visualization_tab()
    
    with tab9:
        optimization_tab()
    
    with tab10:  # ← NUEVO TAB
        sizing_tab()

# ================================================================================================
# EJECUCIÓN
# ================================================================================================

if __name__ == "__main__":
    main()


#C:/Users/Jaume/anaconda3/python.exe -m streamlit run app.py    