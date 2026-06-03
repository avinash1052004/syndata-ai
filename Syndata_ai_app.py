# syndata_ai_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to path to import your module
sys.path.append('.')

# Import your existing classes
from run_syndata_ai import (
    UltimateDataAnalysisAgent, 
    AdvancedVisualizationSuite,
    HTMLReportGenerator,
    ChatAnalyst
)

# Configure the page
st.set_page_config(
    page_title="Syndata AI - Data Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .analysis-progress {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .viz-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

class InMemoryAnalysisAgent(UltimateDataAnalysisAgent):
    """Modified agent that doesn't save files to disk"""
    
    def _create_output_directory(self, dataset_name):
        # Use temporary directory instead of creating permanent folder
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = self.temp_dir
        return self.output_dir
    
    def _save_plot(self, plt_obj, description=""):
        # Don't save plots to disk, store in memory
        self.plot_count += 1
        # Convert plot to base64 for display
        import io
        import base64
        buf = io.BytesIO()
        plt_obj.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt_obj.close()
        
        # Store plot in session state
        if 'plots' not in st.session_state:
            st.session_state.plots = []
        st.session_state.plots.append({
            'description': description,
            'data': plot_data,
            'id': f"plot_{self.plot_count:02d}"
        })
        return f"data:image/png;base64,{plot_data}"

class PDFReportGenerator:
    """Generate PDF reports from analysis results"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def generate_pdf_report(self):
        """Generate a comprehensive PDF report"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.graphics.shapes import Drawing
            from reportlab.graphics.charts.lineplots import LinePlot
            from reportlab.graphics import renderPDF
            import io
            import base64
            
            # Create PDF in memory
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#667eea'),
                alignment=1
            )
            story.append(Paragraph("Syndata AI Analysis Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Dataset Info
            story.append(Paragraph(f"Dataset: {self.agent.dataset_name}", styles['Heading2']))
            story.append(Paragraph(f"Records: {self.agent.data.shape[0]:,}", styles['Normal']))
            story.append(Paragraph(f"Features: {self.agent.data.shape[1]}", styles['Normal']))
            story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Key Insights
            story.append(Paragraph("Key Insights", styles['Heading2']))
            if hasattr(self.agent, 'insights') and self.agent.insights:
                for insight in self.agent.insights[:10]:
                    story.append(Paragraph(f"• {insight}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # ML Results
            if hasattr(self.agent, 'ml_models') and self.agent.ml_models:
                story.append(Paragraph("Machine Learning Results", styles['Heading2']))
                for target, results in self.agent.ml_models.items():
                    story.append(Paragraph(f"Target: {target}", styles['Heading3']))
                    story.append(Paragraph(f"Best Model: {results.get('best_model_name', 'N/A')}", styles['Normal']))
                    story.append(Paragraph(f"Best Score: {results.get('best_score', 'N/A'):.3f}", styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            # Add plots if available
            if 'plots' in st.session_state and st.session_state.plots:
                story.append(Paragraph("Visualizations", styles['Heading2']))
                for plot in st.session_state.plots[:5]:  # Limit to 5 plots in PDF
                    try:
                        # Convert base64 to image
                        img_data = base64.b64decode(plot['data'])
                        img_buffer = io.BytesIO(img_data)
                        story.append(Paragraph(plot['description'], styles['Heading3']))
                        story.append(Image(img_buffer, width=6*inch, height=4*inch))
                        story.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        continue
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return buffer.getvalue()
            
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            return None

class SyndataAIApp:
    def __init__(self):
        self.agent = None
        self.chat_analyst = None
        
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: #667eea;'>🔮 Syndata AI</h2>
            <p style='color: #666;'>Intelligent Data Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 Data Analysis", "🤖 AI Chat", "📈 Visualizations", "📄 PDF Report"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Actions")
        
        if st.sidebar.button("🔄 New Analysis"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
        # Show analysis progress if available
        if 'analysis_progress' in st.session_state:
            st.sidebar.markdown("### 📊 Analysis Progress")
            progress = st.session_state.analysis_progress
            st.sidebar.progress(progress['value'], text=progress['text'])
            
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style='text-align: center; color: #888;'>
            <small>AI Powered Advanced Analytics</small>
        </div>
        """, unsafe_allow_html=True)
        
        return page
    
    def update_progress(self, value, text):
        """Update analysis progress"""
        st.session_state.analysis_progress = {'value': value, 'text': text}
    
    def run_comprehensive_analysis(self, target_column=None):
        """Run comprehensive analysis on all columns without saving files"""
        try:
            self.update_progress(0.1, "🔄 Initializing analysis...")
            
            # Run the main comprehensive analysis
            analysis_results = self.agent.run_comprehensive_analysis(target_column)
            
            self.update_progress(0.6, "📊 Generating interactive visualizations...")
            
            # Generate advanced visualizations in memory
            viz_suite = AdvancedVisualizationSuite(self.agent)
            
            # Store visualization data in session state
            if 'visualizations' not in st.session_state:
                st.session_state.visualizations = {}
            
            # Generate and store 3D visualizations
            st.session_state.visualizations['3d'] = viz_suite.create_3d_visualizations(target_column)
            
            self.update_progress(0.7, "🔍 Creating correlation analysis...")
            st.session_state.visualizations['correlation'] = viz_suite.create_interactive_correlation_explorer()
            
            self.update_progress(0.8, "📈 Generating multi-dimensional analysis...")
            st.session_state.visualizations['multi_dim'] = viz_suite.create_multi_dimensional_analysis()
            
            self.update_progress(0.85, "📊 Creating statistical plots...")
            st.session_state.visualizations['statistical'] = viz_suite.create_advanced_statistical_plots()
            
            self.update_progress(0.9, "💡 Generating insights...")
            
            # Store results in session state
            st.session_state.analysis_results = analysis_results
            st.session_state.target_column = target_column
            st.session_state.analysis_complete = True
            
            self.update_progress(1.0, "✅ Analysis complete!")
            
            return analysis_results
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            self.update_progress(0.0, "❌ Analysis failed")
            return None
    
    def display_plotly_chart(self, html_content):
        """Display Plotly chart from HTML content"""
        try:
            from streamlit.components.v1 import html
            html(html_content, height=600, scrolling=False)
        except Exception as e:
            st.error(f"Error displaying chart: {str(e)}")
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.markdown('<div class="main-header">Syndata AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Transform Data into Intelligent Decisions</div>', unsafe_allow_html=True)
        
        # Hero section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 AI-Powered Data Intelligence
            
            Unlock the power of artificial intelligence and machine learning to:
            - **Discover hidden patterns** in your data
            - **Generate actionable insights** automatically
            - **Create stunning 3D visualizations**
            - **Build predictive models** with one click
            - **Chat with your data** in natural language
            
            Upload your dataset and let Syndata AI do the magic! ✨
            """)
            
            # File upload section
            st.markdown("### 📁 Get Started")
            uploaded_file = st.file_uploader(
                "Upload your dataset", 
                type=['csv', 'xlsx', 'json'],
                help="Supported formats: CSV, Excel, JSON"
            )
            
            if uploaded_file is not None:
                # Display file info
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.1f} KB",
                    "File type": uploaded_file.type
                }
                st.write(file_details)
                
                if st.button("🚀 Launch Comprehensive Analysis", use_container_width=True):
                    with st.spinner("🧠 Syndata AI is analyzing your data..."):
                        try:
                            # Use in-memory agent
                            self.agent = InMemoryAnalysisAgent("Syndata AI Analyst")
                            
                            # Load data
                            self.agent.load_data(uploaded_file)
                            st.session_state.agent = self.agent
                            st.session_state.file_name = uploaded_file.name
                            
                            # Initialize progress
                            st.session_state.analysis_progress = {'value': 0, 'text': 'Starting analysis...'}
                            st.session_state.analysis_complete = False
                            
                            # Run comprehensive analysis
                            analysis_results = self.run_comprehensive_analysis()
                            
                            if analysis_results:
                                st.success("✅ Comprehensive analysis completed successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Analysis failed")
                                
                        except Exception as e:
                            st.error(f"❌ Error loading data: {str(e)}")
            
        with col2:
            st.markdown("### 🎯 Key Features")
            
            features = [
                "🤖 Automated Machine Learning",
                "📊 Interactive 3D Visualizations",
                "🔍 Advanced Correlation Analysis",
                "📈 Time Series Forecasting",
                "👥 Customer Segmentation",
                "💬 Natural Language Chat",
                "📋 In-Memory Analysis",
                "🎯 Predictive Analytics"
            ]
            
            for feature in features:
                st.markdown(f'<div class="feature-card">{feature}</div>', unsafe_allow_html=True)
        
        # Sample data section
        st.markdown("---")
        st.markdown("### 🎮 Try with Sample Data")
        
        if st.button("🎲 Load Sample Dataset", use_container_width=True):
            with st.spinner("Loading sample data..."):
                try:
                    # Use in-memory agent
                    self.agent = InMemoryAnalysisAgent("Syndata AI Analyst")
                    self.agent.load_data(None)  # This creates sample data
                    st.session_state.agent = self.agent
                    st.session_state.file_name = "sample_data.csv"
                    
                    # Initialize progress
                    st.session_state.analysis_progress = {'value': 0, 'text': 'Starting analysis...'}
                    st.session_state.analysis_complete = False
                    
                    # Run comprehensive analysis
                    analysis_results = self.run_comprehensive_analysis()
                    
                    if analysis_results:
                        st.success("✅ Comprehensive analysis completed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed")
                        
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
    
    def render_data_analysis(self):
        """Render the data analysis page"""
        st.markdown("## 📊 Data Analysis Center")
        
        if 'agent' not in st.session_state:
            st.warning("⚠️ Please load a dataset first from the Dashboard page.")
            return
        
        self.agent = st.session_state.agent
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Rows", f"{self.agent.data.shape[0]:,}")
        with col2:
            st.metric("📊 Columns", self.agent.data.shape[1])
        with col3:
            numeric_cols = len(self.agent.data.select_dtypes(include=[np.number]).columns)
            st.metric("🔍 Numeric Columns", numeric_cols)
        with col4:
            text_cols = len(self.agent.data.select_dtypes(include=['object']).columns)
            st.metric("📝 Text Columns", text_cols)
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(self.agent.data.head(10), use_container_width=True)
        
        # Data statistics
        st.subheader("📈 Data Statistics")
        st.dataframe(self.agent.data.describe(), use_container_width=True)
        
        # Analysis configuration
        st.subheader("🎯 Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Select Target Column (for ML - Optional)",
                options=["None"] + list(self.agent.data.columns),
                help="Select the column you want to predict (optional)"
            )
            if target_column == "None":
                target_column = None
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Depth",
                ["Comprehensive Analysis", "Quick Analysis", "ML Focus", "Visualization Focus"]
            )
        
        # Run analysis button
        if st.button("🚀 Run Comprehensive Analysis", use_container_width=True, type="primary"):
            with st.spinner("🧠 Syndata AI is performing advanced analysis..."):
                try:
                    # Initialize progress
                    st.session_state.analysis_progress = {'value': 0, 'text': 'Starting analysis...'}
                    st.session_state.analysis_complete = False
                    
                    # Run comprehensive analysis
                    analysis_results = self.run_comprehensive_analysis(target_column)
                    
                    if analysis_results:
                        st.success("🎉 Comprehensive analysis completed successfully!")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
        
        # Show analysis results if available
        if st.session_state.get('analysis_complete', False):
            st.subheader("📋 Analysis Summary")
            
            # Key Insights
            if hasattr(self.agent, 'insights') and self.agent.insights:
                st.markdown("#### 💡 Key Insights")
                insights_container = st.container()
                with insights_container:
                    for insight in self.agent.insights[:15]:
                        st.write(f"• {insight}")
            
            # ML Results
            if hasattr(self.agent, 'ml_models') and self.agent.ml_models:
                st.markdown("#### 🤖 Machine Learning Results")
                for target, results in self.agent.ml_models.items():
                    with st.expander(f"Model for {target}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best Model", results.get('best_model_name', 'N/A'))
                        with col2:
                            st.metric("Best Score", f"{results.get('best_score', 0):.3f}")
                        with col3:
                            st.metric("Problem Type", results.get('problem_type', 'N/A'))
            
            # Time Series Results
            if hasattr(self.agent, 'time_series_columns') and self.agent.time_series_columns:
                st.markdown("#### ⏰ Time Series Analysis")
                st.write(f"Detected time series columns: {', '.join(self.agent.time_series_columns)}")
    
    def render_ai_chat(self):
        """Render the AI chat interface"""
        st.markdown("## 🤖 AI Data Chat Assistant")
        
        if 'agent' not in st.session_state:
            st.warning("⚠️ Please load a dataset first from the Dashboard page.")
            return
        
        self.agent = st.session_state.agent
        target_column = st.session_state.get('target_column', None)
        
        # Initialize chat analyst
        if 'chat_analyst' not in st.session_state:
            st.session_state.chat_analyst = ChatAnalyst(self.agent, target_column)
            st.session_state.chat_history = []
        
        self.chat_analyst = st.session_state.chat_analyst
        
        # Chat examples
        st.markdown("""
        ### 💡 Example Questions:
        - "What's the average income in the data?"
        - "Show correlation between age and purchase amount"
        - "What are the unique regions?"
        - "Run prediction for customer churn with age=30, income=50000"
        - "Show feature importance for the model"
        - "What are the main insights from the data?"
        - "Show me the data distribution for sales"
        """)
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class='chat-message user-message'>
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Format AI response with line breaks
                    formatted_content = message['content'].replace('\n', '<br>')
                    st.markdown(f"""
                    <div class='chat-message ai-message'>
                        <strong>Syndata AI:</strong> {formatted_content}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., 'What's the correlation between age and income?'",
                key="chat_input"
            )
        
        with col2:
            send_button = st.button("Send", use_container_width=True, key="send_chat")
        
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                try:
                    response = self.chat_analyst.answer(user_input)
                    
                    # Add AI response to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    def render_visualizations(self):
        """Render the visualizations page"""
        st.markdown("## 📈 Interactive Visualizations")
        
        if 'agent' not in st.session_state:
            st.warning("⚠️ Please load a dataset first from the Dashboard page.")
            return
        
        if not st.session_state.get('analysis_complete', False):
            st.warning("⚠️ Please run analysis first to generate visualizations.")
            return
        
        self.agent = st.session_state.agent
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["All Visualizations", "3D Plots", "Correlation Matrix", "Statistical Plots", "Data Distributions"]
        )
        
        # Display static plots
        if 'plots' in st.session_state and st.session_state.plots:
            st.markdown("### 📊 Static Visualizations")
            for plot in st.session_state.plots:
                st.markdown(f"#### {plot['description'].replace('_', ' ').title()}")
                st.image(f"data:image/png;base64,{plot['data']}", use_column_width=True)
                st.markdown("---")
        
        # Display interactive visualizations if available
        if 'visualizations' in st.session_state:
            viz_data = st.session_state.visualizations
            
            if viz_type in ["All Visualizations", "3D Plots"] and '3d' in viz_data:
                st.markdown("### 🎯 3D Visualizations")
                for viz in viz_data['3d']:
                    if os.path.exists(viz):
                        with open(viz, 'r') as f:
                            html_content = f.read()
                            self.display_plotly_chart(html_content)
            
            if viz_type in ["All Visualizations", "Correlation Matrix"] and 'correlation' in viz_data:
                st.markdown("### 🔗 Correlation Matrix")
                if os.path.exists(viz_data['correlation']):
                    with open(viz_data['correlation'], 'r') as f:
                        html_content = f.read()
                        self.display_plotly_chart(html_content)
    
    def render_pdf_report(self):
        """Render the PDF report generation page"""
        st.markdown("## 📄 PDF Analysis Report")
        
        if 'agent' not in st.session_state:
            st.warning("⚠️ Please load a dataset and run analysis first.")
            return
        
        if not st.session_state.get('analysis_complete', False):
            st.warning("⚠️ Please run analysis first to generate report content.")
            return
        
        self.agent = st.session_state.agent
        
        st.markdown("""
        ### Generate Comprehensive PDF Report
        
        Create a professional PDF report containing:
        - Dataset overview and statistics
        - Key insights and findings
        - Machine learning results
        - Visualizations and charts
        - Recommendations and next steps
        """)
        
        # Report customization
        st.subheader("🔧 Report Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_ml = st.checkbox("Include ML Results", value=True)
            include_viz = st.checkbox("Include Visualizations", value=True)
        
        with col2:
            include_insights = st.checkbox("Include Insights", value=True)
            include_stats = st.checkbox("Include Statistics", value=True)
        
        # Generate PDF button
        if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("🔄 Generating PDF report..."):
                try:
                    pdf_generator = PDFReportGenerator(self.agent)
                    pdf_data = pdf_generator.generate_pdf_report()
                    
                    if pdf_data:
                        # Create download button
                        st.success("✅ PDF report generated successfully!")
                        
                        # Convert to base64 for download
                        b64 = base64.b64encode(pdf_data).decode()
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="syndata_ai_analysis_report.pdf">📥 Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Display PDF preview
                        st.subheader("📋 Report Preview")
                        st.info("""
                        **Report Contents:**
                        - Executive Summary
                        - Dataset Overview
                        - Key Insights
                        - Machine Learning Results
                        - Visualizations
                        - Recommendations
                        """)
                    else:
                        st.error("❌ Failed to generate PDF report")
                        
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
        
        # Show analysis summary
        if st.session_state.get('analysis_complete', False):
            st.subheader("📊 Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Visualizations", len(st.session_state.get('plots', [])))
                st.metric("ML Models Trained", len(getattr(self.agent, 'ml_models', {})))
            
            with col2:
                st.metric("Key Insights", len(getattr(self.agent, 'insights', [])))
                st.metric("Time Series Columns", len(getattr(self.agent, 'time_series_columns', [])))
    
    def run(self):
        """Main application runner"""
        # Initialize session state
        if 'agent' not in st.session_state:
            st.session_state.agent = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'analysis_progress' not in st.session_state:
            st.session_state.analysis_progress = {'value': 0, 'text': ''}
        if 'plots' not in st.session_state:
            st.session_state.plots = []
        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = {}
        
        # Render sidebar and get current page
        page = self.render_sidebar()
        
        # Render the selected page
        if page == "🏠 Dashboard":
            self.render_dashboard()
        elif page == "📊 Data Analysis":
            self.render_data_analysis()
        elif page == "🤖 AI Chat":
            self.render_ai_chat()
        elif page == "📈 Visualizations":
            self.render_visualizations()
        elif page == "📄 PDF Report":
            self.render_pdf_report()

# Run the app
if __name__ == "__main__":
    app = SyndataAIApp()
    app.run()
