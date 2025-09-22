"""
Qready Integration Module - Metodología de análisis de bienestar laboral
Basado en los documentos de Qready y Quirón Prevención analizados
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class QreadyWellbeingAnalyzer:
    """Analizador de bienestar laboral basado en metodología Qready"""
    
    def __init__(self):
        self.wellbeing_dimensions = {
            'Bienestar (positividad)': {
                'questions': [
                    'En las últimas 2 semanas me he sentido alegre y con buen ánimo',
                    'Me siento satisfecho/a con mi vida',
                    'Tengo energía para realizar mis actividades diarias'
                ],
                'scale': 'Nunca;Raramente;Algunas veces;A menudo;Siempre'
            },
            'Estrés': {
                'questions': [
                    'Me siento abrumado/a por las demandas del trabajo',
                    'Tengo dificultades para relajarme después del trabajo',
                    'Me preocupo constantemente por temas laborales'
                ],
                'scale': 'Nunca;Casi nunca;A veces;A menudo;Muy a menudo'
            },
            'Compromiso Laboral': {
                'questions': [
                    'Me siento comprometido/a con mi trabajo',
                    'Encuentro significado en lo que hago',
                    'Me siento valorado/a en mi organización'
                ],
                'scale': 'Totalmente en desacuerdo;En desacuerdo;Neutral;De acuerdo;Totalmente de acuerdo'
            },
            'Salud Física': {
                'questions': [
                    'Duermo lo suficiente para sentirme descansado/a',
                    'Mantengo hábitos alimentarios saludables',
                    'Realizo ejercicio físico regularmente'
                ],
                'scale': 'Nunca;Raramente;Algunas veces;A menudo;Siempre'
            }
        }
    
    def generate_wellbeing_assessment(self):
        """Genera una evaluación de bienestar basada en Qready"""
        assessment = {}
        for dimension, data in self.wellbeing_dimensions.items():
            assessment[dimension] = {
                'questions': data['questions'],
                'responses': np.random.choice([1, 2, 3, 4, 5], size=len(data['questions'])),
                'scale_labels': data['scale'].split(';')
            }
        return assessment
    
    def calculate_dimension_scores(self, assessment):
        """Calcula puntuaciones por dimensión"""
        scores = {}
        for dimension, data in assessment.items():
            scores[dimension] = {
                'raw_score': np.mean(data['responses']),
                'percentage': (np.mean(data['responses']) - 1) / 4 * 100,  # Normalizar a 0-100%
                'responses': data['responses']
            }
        return scores
    
    def generate_wellbeing_visualization(self, scores):
        """Genera visualización de bienestar estilo Qready"""
        dimensions = list(scores.keys())
        percentages = [scores[dim]['percentage'] for dim in dimensions]
        
        # Radar chart para visualizar dimensiones
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=percentages,
            theta=dimensions,
            fill='toself',
            name='Puntuación Bienestar',
            line=dict(color='rgb(46, 139, 87)'),
            fillcolor='rgba(46, 139, 87, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Perfil de Bienestar Laboral (Metodología Qready)",
            height=500
        )
        
        return fig
    
    def generate_risk_indicators(self, scores):
        """Genera indicadores de riesgo basados en puntuaciones"""
        risk_indicators = []
        
        for dimension, score_data in scores.items():
            percentage = score_data['percentage']
            
            if percentage < 40:
                risk_level = "Alto Riesgo"
                color = "red"
                recommendation = f"Se recomienda intervención inmediata en {dimension}"
            elif percentage < 60:
                risk_level = "Riesgo Moderado" 
                color = "orange"
                recommendation = f"Monitoreo y mejoras en {dimension}"
            else:
                risk_level = "Bajo Riesgo"
                color = "green" 
                recommendation = f"Mantener nivel actual de {dimension}"
            
            risk_indicators.append({
                'dimension': dimension,
                'percentage': percentage,
                'risk_level': risk_level,
                'color': color,
                'recommendation': recommendation
            })
        
        return risk_indicators
    
    def predict_absenteeism_risk(self, scores):
        """Predice riesgo de absentismo basado en puntuaciones de bienestar"""
        # Modelo simple basado en la metodología Qready
        stress_score = scores.get('Estrés', {}).get('percentage', 50)
        wellbeing_score = scores.get('Bienestar (positividad)', {}).get('percentage', 50)
        health_score = scores.get('Salud Física', {}).get('percentage', 50)
        
        # Invertir puntuación de estrés (más estrés = mayor riesgo)
        stress_risk = 100 - stress_score
        
        # Calcular riesgo combinado
        absenteeism_risk = (stress_risk * 0.4 + (100 - wellbeing_score) * 0.3 + (100 - health_score) * 0.3)
        
        if absenteeism_risk > 70:
            risk_category = "Alto"
            risk_color = "red"
        elif absenteeism_risk > 50:
            risk_category = "Medio"
            risk_color = "orange"
        else:
            risk_category = "Bajo"
            risk_color = "green"
        
        return {
            'risk_percentage': absenteeism_risk,
            'risk_category': risk_category,
            'color': risk_color,
            'factors': {
                'Estrés': stress_risk,
                'Bienestar': 100 - wellbeing_score,
                'Salud Física': 100 - health_score
            }
        }