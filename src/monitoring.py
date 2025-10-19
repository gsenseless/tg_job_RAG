from google.cloud import firestore
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict
import streamlit as st


class MonitoringSubsystem:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.db = firestore.Client(project=project_id, database='ragdb')
    
    def log_query(self, user_id: str, num_results: int, avg_similarity: float):
        """Log query event."""
        self.db.collection("queries").add({
            "user_id": user_id,
            "num_results": num_results,
            "avg_similarity": avg_similarity,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    
    def log_feedback(self, user_id: str, job_id: str, liked: bool):
        """Store user feedback."""
        self.db.collection("feedback").add({
            "user_id": user_id,
            "job_id": job_id,
            "liked": liked,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    
    def get_queries_df(self) -> pd.DataFrame:
        """Retrieve query logs as DataFrame."""
        docs = self.db.collection("queries").stream()
        data = [doc.to_dict() for doc in docs]
        if not data:
            return pd.DataFrame(columns=["user_id", "num_results", "avg_similarity", "timestamp"])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    
    def get_feedback_df(self) -> pd.DataFrame:
        """Retrieve feedback logs as DataFrame."""
        docs = self.db.collection("feedback").stream()
        data = [doc.to_dict() for doc in docs]
        if not data:
            return pd.DataFrame(columns=["user_id", "job_id", "liked", "timestamp"])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    
    def render_dashboard(self):
        """Render monitoring dashboard focused on likes/dislikes."""
        st.title("📊 Job Matching System Dashboard")
        
        feedback_df = self.get_feedback_df()
        queries_df = self.get_queries_df()
        
        has_feedback = len(feedback_df) > 0
        has_queries = len(queries_df) > 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", len(queries_df) if has_queries else 0)
        with col2:
            st.metric("Total Feedback", len(feedback_df) if has_feedback else 0)
        with col3:
            if has_feedback:
                likes = int(feedback_df['liked'].sum())
                st.metric("👍 Likes", likes)
            else:
                st.metric("👍 Likes", 0)
        with col4:
            if has_feedback:
                dislikes = int((~feedback_df['liked']).sum())
                st.metric("👎 Dislikes", dislikes)
            else:
                st.metric("👎 Dislikes", 0)
        
        st.divider()
        
        if has_feedback:
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Likes vs Dislikes")
                likes = int(feedback_df["liked"].sum())
                dislikes = len(feedback_df) - likes
                
                fig1 = go.Figure(data=[go.Pie(
                    labels=['👍 Likes', '👎 Dislikes'],
                    values=[likes, dislikes],
                    marker=dict(colors=['#28a745', '#dc3545']),
                    hole=0.3
                )])
                fig1.update_layout(
                    title="Feedback Distribution",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_right:
                st.subheader("📈 Feedback Over Time")
                feedback_daily = feedback_df.copy()
                feedback_daily['date'] = feedback_daily['timestamp'].dt.date
                
                daily_stats = feedback_daily.groupby('date').agg({
                    'liked': ['sum', 'count']
                }).reset_index()
                daily_stats.columns = ['date', 'likes', 'total']
                daily_stats['dislikes'] = daily_stats['total'] - daily_stats['likes']
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=daily_stats['date'], y=daily_stats['likes'],
                    name='Likes', mode='lines+markers',
                    line=dict(color='#28a745', width=3),
                    marker=dict(size=8)
                ))
                fig2.add_trace(go.Scatter(
                    x=daily_stats['date'], y=daily_stats['dislikes'],
                    name='Dislikes', mode='lines+markers',
                    line=dict(color='#dc3545', width=3),
                    marker=dict(size=8)
                ))
                fig2.update_layout(
                    title="Daily Likes & Dislikes Trend",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("📊 Satisfaction Rate Over Time")
            satisfaction_daily = daily_stats.copy()
            satisfaction_daily['rate'] = (satisfaction_daily['likes'] / satisfaction_daily['total'] * 100).round(1)
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=satisfaction_daily['date'],
                y=satisfaction_daily['rate'],
                marker=dict(
                    color=satisfaction_daily['rate'],
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=100,
                    showscale=True,
                    colorbar=dict(title="Rate %")
                ),
                text=satisfaction_daily['rate'].astype(str) + '%',
                textposition='outside'
            ))
            fig3.update_layout(
                title="Daily Satisfaction Rate (% Likes)",
                xaxis_title="Date",
                yaxis_title="Satisfaction Rate (%)",
                yaxis=dict(range=[0, 110]),
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            st.subheader("👤 Most Active Users (Feedback)")
            user_feedback = feedback_df.groupby('user_id').agg({
                'liked': ['sum', 'count']
            }).reset_index()
            user_feedback.columns = ['user_id', 'likes', 'total']
            user_feedback['dislikes'] = user_feedback['total'] - user_feedback['likes']
            user_feedback = user_feedback.sort_values('total', ascending=False).head(10)
            
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                name='Likes',
                x=user_feedback['user_id'],
                y=user_feedback['likes'],
                marker_color='#28a745'
            ))
            fig4.add_trace(go.Bar(
                name='Dislikes',
                x=user_feedback['user_id'],
                y=user_feedback['dislikes'],
                marker_color='#dc3545'
            ))
            fig4.update_layout(
                title="Top 10 Users by Feedback Volume",
                xaxis_title="User ID",
                yaxis_title="Count",
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("⚠️ No feedback data available yet. Start using the Job Search tab to generate feedback data!")
        
        if has_queries:
            st.divider()
            st.subheader("🎯 Similarity Score Distribution")
            if 'avg_similarity' in queries_df.columns:
                fig7 = px.histogram(queries_df, x="avg_similarity", nbins=30,
                                   title="Average Similarity Scores",
                                   labels={"avg_similarity": "Similarity Score"})
                fig7.update_traces(marker_color='#17a2b8')
                fig7.update_layout(height=400)
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.info("No similarity data available")
