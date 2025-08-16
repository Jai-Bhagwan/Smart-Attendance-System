import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, date
from PIL import Image
import io

# Initialize session state
if 'face_encodings' not in st.session_state:
    st.session_state.face_encodings = {}
if 'student_names' not in st.session_state:
    st.session_state.student_names = []
if 'attendance_today' not in st.session_state:
    st.session_state.attendance_today = set()

# File paths
STUDENTS_CSV = "students.csv"
ATTENDANCE_CSV = "attendance.csv"
ENCODINGS_FILE = "face_encodings.pkl"

def load_data():
    """Load existing student data and face encodings"""
    # Load students CSV
    if os.path.exists(STUDENTS_CSV):
        students_df = pd.read_csv(STUDENTS_CSV)
        st.session_state.student_names = students_df['name'].tolist()
    else:
        # Create empty students CSV
        pd.DataFrame(columns=['name', 'student_id', 'registration_date']).to_csv(STUDENTS_CSV, index=False)
    
    # Load attendance CSV
    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=['name', 'student_id', 'date', 'time', 'status']).to_csv(ATTENDANCE_CSV, index=False)
    
    # Load face encodings
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            st.session_state.face_encodings = pickle.load(f)

def save_face_encodings():
    """Save face encodings to file"""
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(st.session_state.face_encodings, f)

def detect_faces(image):
    """Detect faces using face_recognition"""
    # Convert PIL image to RGB array if needed
    if isinstance(image, Image.Image):
        image_rgb = np.array(image)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find all face locations
    face_locations = face_recognition.face_locations(image_rgb)
    
    # Get face encodings for recognized faces
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
    
    return face_locations, face_encodings

def mark_attendance(name, student_id):
    """Mark attendance for a student"""
    today = date.today().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Check if already marked today
    attendance_key = f"{name}_{today}"
    if attendance_key in st.session_state.attendance_today:
        return False, "Attendance already marked for today"
    
    # Add to today's attendance
    st.session_state.attendance_today.add(attendance_key)
    
    # Save to CSV
    new_attendance = {
        'name': name,
        'student_id': student_id,
        'date': today,
        'time': current_time,
        'status': 'Present'
    }
    
    attendance_df = pd.read_csv(ATTENDANCE_CSV)
    attendance_df = pd.concat([attendance_df, pd.DataFrame([new_attendance])], ignore_index=True)
    attendance_df.to_csv(ATTENDANCE_CSV, index=False)
    
    return True, f"Attendance marked for {name} at {current_time}"



def main():
    st.title("🎓 Facial Recognition Attendance System")
    st.markdown("---")
    
    # Load data on startup
    load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📝 Register Student", 
        "📹 Take Attendance", 
        "👥 View Students", 
        "📊 Attendance Reports"
    ])
    
    if page == "📝 Register Student":
        st.header("Register New Student")
        
        st.info("Students will be registered when they first appear in front of the camera during attendance.")
        
        name = st.text_input("Student Name", placeholder="Enter full name")
        student_id = st.text_input("Student ID", placeholder="Enter student ID")
        
        if st.button("Register Student", type="primary"):
            if name and student_id:
                # Check if student already exists
                if name in st.session_state.student_names:
                    st.error("Student with this name already exists!")
                else:
                    # Register student without face encoding initially
                    st.session_state.student_names.append(name)
                    
                    # Save to CSV
                    new_student = {
                        'name': name,
                        'student_id': student_id,
                        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    students_df = pd.read_csv(STUDENTS_CSV)
                    students_df = pd.concat([students_df, pd.DataFrame([new_student])], ignore_index=True)
                    students_df.to_csv(STUDENTS_CSV, index=False)
                    
                    st.success(f"Student {name} registered successfully! Face will be learned during first attendance.")
                    st.rerun()
            else:
                st.error("Please fill in all fields")
    
    elif page == "📹 Take Attendance":
        st.header("Real-time Attendance System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Camera Feed")
            
            # Camera controls
            start_camera = st.button("Start Camera", type="primary")
            stop_camera = st.button("Stop Camera")
            
            # Placeholder for camera feed
            camera_placeholder = st.empty()
            
            # Add manual attendance marking option
            st.subheader("Manual Attendance")
            students_df = pd.read_csv(STUDENTS_CSV)
            if not students_df.empty:
                selected_student = st.selectbox(
                    "Select student for manual attendance",
                    [""] + students_df['name'].tolist()
                )
                if st.button("Mark Present") and selected_student:
                    student_row = students_df[students_df['name'] == selected_student]
                    if not student_row.empty:
                        student_id = student_row.iloc[0]['student_id']
                        success, message = mark_attendance(selected_student, student_id)
                        if success:
                            st.success(message)
                        else:
                            st.warning(message)
            else:
                st.info("No students registered yet. Please register students first.")
            
            if start_camera:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not access camera. Please check camera permissions.")
                else:
                    st.success("Camera started! Face detection is running.")
                    
                    # Get students data for attendance marking
                    students_df = pd.read_csv(STUDENTS_CSV)
                    
                    frame_count = 0
                    recognition_interval = 30  # Recognize every 30 frames
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame from camera")
                            break
                        
                        frame_count += 1
                        
                        # Perform face detection every N frames to improve performance
                        if frame_count % recognition_interval == 0:
                            face_locations, face_encodings = detect_faces(frame)
                            
                            # Draw rectangles around detected faces
                            for (top, right, bottom, left) in face_locations:
                                # Draw rectangle around face
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                
                                # Get face encoding and compare with known faces
                                face_encoding = face_encodings[face_locations.index((top, right, bottom, left))]
                                
                                # Compare with known face encodings
                                matches = []
                                for known_name, known_encoding in st.session_state.face_encodings.items():
                                    match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
                                    if match:
                                        matches.append(known_name)
                                
                                if matches:
                                    name = matches[0]
                                    # Mark attendance for matched student
                                    students_df = pd.read_csv(STUDENTS_CSV)
                                    student_row = students_df[students_df['name'] == name]
                                    if not student_row.empty:
                                        student_id = student_row.iloc[0]['student_id']
                                        success, message = mark_attendance(name, student_id)
                                else:
                                    name = "Unknown"
                                
                                # Draw name label
                                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                                cv2.putText(frame, name, (left + 6, bottom - 6), 
                                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        
                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Check if stop button was pressed
                        if stop_camera:
                            break
                    
                    cap.release()
        
        with col2:
            st.subheader("Today's Attendance")
            
            # Load today's attendance
            attendance_df = pd.read_csv(ATTENDANCE_CSV)
            today = date.today().strftime('%Y-%m-%d')
            today_attendance = attendance_df[attendance_df['date'] == today]
            
            if not today_attendance.empty:
                st.dataframe(
                    today_attendance[['name', 'time', 'status']], 
                    use_container_width=True,
                    hide_index=True
                )
                st.metric("Students Present Today", len(today_attendance))
            else:
                st.info("No attendance recorded today yet.")
    
    elif page == "👥 View Students":
        st.header("Registered Students")
        
        students_df = pd.read_csv(STUDENTS_CSV)
        
        if not students_df.empty:
            st.dataframe(
                students_df[['name', 'student_id', 'registration_date']], 
                use_container_width=True,
                hide_index=True
            )
            st.metric("Total Registered Students", len(students_df))
            
            # Option to remove student
            st.subheader("Remove Student")
            student_to_remove = st.selectbox(
                "Select student to remove", 
                [""] + students_df['name'].tolist()
            )
            
            if student_to_remove and st.button("Remove Student", type="secondary"):
                # Remove from dataframe
                students_df = students_df[students_df['name'] != student_to_remove]
                students_df.to_csv(STUDENTS_CSV, index=False)
                
                # Remove from session state
                if student_to_remove in st.session_state.face_encodings:
                    del st.session_state.face_encodings[student_to_remove]
                    save_face_encodings()
                
                if student_to_remove in st.session_state.student_names:
                    st.session_state.student_names.remove(student_to_remove)
                
                st.success(f"Student {student_to_remove} removed successfully!")
                st.rerun()
        else:
            st.info("No students registered yet.")
    
    elif page == "📊 Attendance Reports":
        st.header("Attendance Reports")
        
        attendance_df = pd.read_csv(ATTENDANCE_CSV)
        
        if not attendance_df.empty:
            # Date range selector
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input("Start Date", value=date.today())
            with col2:
                end_date = st.date_input("End Date", value=date.today())
            
            # Filter data by date range
            filtered_df = attendance_df[
                (pd.to_datetime(attendance_df['date']).dt.date >= start_date) &
                (pd.to_datetime(attendance_df['date']).dt.date <= end_date)
            ]
            
            if not filtered_df.empty:
                st.subheader("Attendance Records")
                st.dataframe(
                    filtered_df[['name', 'student_id', 'date', 'time', 'status']], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(filtered_df))
                
                with col2:
                    unique_students = filtered_df['name'].nunique()
                    st.metric("Unique Students", unique_students)
                
                with col3:
                    unique_dates = filtered_df['date'].nunique()
                    st.metric("Days Covered", unique_dates)
                
                # Attendance by student
                st.subheader("Attendance Summary by Student")
                student_summary = filtered_df.groupby('name').agg({
                    'date': 'count',
                    'student_id': 'first'
                }).rename(columns={'date': 'days_present'}).reset_index()
                
                st.dataframe(
                    student_summary[['name', 'student_id', 'days_present']], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Attendance Report (CSV)",
                    data=csv_data,
                    file_name=f"attendance_report_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No attendance records found for the selected date range.")
        else:
            st.info("No attendance records available yet.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Make sure your camera is working and you have proper lighting for best face recognition results.")

if __name__ == "__main__":
    main()