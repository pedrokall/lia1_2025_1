import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
import queue
import pandas as pd
from personal_ai import *
from time import time, sleep
import os

st.set_page_config(
    layout="wide"
)

# Listar vídeos disponíveis na pasta 'videos'
video_dir = 'videos'
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mov', '.mp4', '.avi'))]
selected_video = st.sidebar.selectbox('Selecione o vídeo', video_files)

# Definir a fonte de vídeo
video_source = os.path.join(video_dir, selected_video)

personalAI = PersonalAI(video_source)
personalAI.run()

st.sidebar.title("AI Personal Trainer")
display_charts = st.sidebar.checkbox('Display charts', value=True)
reset = st.sidebar.button("Reset")

rotation_option = st.sidebar.selectbox(
    "Rotação do vídeo",
    ("0°", "90° Horário", "180°", "90° Anti-horário"),
    index=1  # padrão: 90° Horário
)

col1, col2 = st.columns(2)
# run = st.sidebar.checkbox('Ligar Webcam')
# landmarks = st.sidebar.checkbox('Mapeamento facial')

frame, landmarks, ts = personalAI.image_q.get()
df_nodes_x = pd.DataFrame()
df_nodes_y = pd.DataFrame()

video_container = col1.empty()
status_container = col2.empty()
chart_container = col2.empty()

c = 0
status = "standby"
useful_points = [11, 12, 23, 24, 25, 26, 27, 28]

count = 0
direction = "down"
last_state = "standby"

while True:
    frame, landmarks, ts = personalAI.image_q.get()
    if ts == "done": break

    if len(landmarks.pose_landmarks) > 0:
        frame, knee_angle = personalAI.draw_angle(frame, landmarks, 23, 25, 27)  # quadril-joelho-tornozelo (perna esquerda)
        frame, hip_angle = personalAI.draw_angle(frame, landmarks, 11, 23, 25)   # ombro-quadril-joelho (flexão do quadril)
        frame, back_angle = personalAI.draw_angle(frame, landmarks, 12, 11, 23)  # ombro-ombro-quadril (inclinação das costas)

        df_y = pd.DataFrame([i.y for i in [i for i in landmarks.pose_landmarks[0]]]).rename(columns={0: ts}).transpose()
        df_nodes_y = pd.concat([df_nodes_y, df_y])
        
        knee_abs = abs(knee_angle)
        hip_abs = abs(hip_angle)
        
        debug_info = f"Knee: {knee_abs:.1f}° | Hip: {hip_abs:.1f}° | Status: {status} | Dir: {direction}"
        
        if knee_abs > 140 and hip_abs > 140:
            if status != "ready":
                status = "ready"
                direction = "down"

        if status == "ready":
            if direction == "down" and (knee_abs < 90 or hip_abs < 90):
                direction = "up"
                status = "descending"
                
            # Fase de subida: retorno à posição inicial
        elif status == "descending":
            if direction == "up" and knee_abs > 130 and hip_abs > 130:
                direction = "down"
                status = "ready"
                count += 1

        # Aplicar rotação conforme seleção do usuário
        if rotation_option == "90° Horário":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_option == "180°":
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_option == "90° Anti-horário":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Se for "0°", não faz nada
        video_container.image(frame)
        
        # Atualizar status sempre
        if status == "standby":
            status_m = f":orange[{status}]"
        elif status == "ready":
            status_m = f":green[{status}]"
        else:
            status_m = f":blue[{status}]"
            
        status_container.markdown(f"""
        ### **Status:** {status_m}
        ### Deadlifts: {int(count)}
        ### Knee Angle: {int(knee_abs)}°
        ### Hip Angle: {int(hip_abs)}°
        ### Back Angle: {int(abs(back_angle))}°
        
        **Debug:** {debug_info}
        
        ---
        """)
        
        # Atualizar gráfico apenas ocasionalmente
        if c % 30 == 0 and display_charts and len(df_nodes_y) > 10:  # A cada 30 frames
            chart_container.line_chart(df_nodes_y[useful_points])
        
        c += 1
