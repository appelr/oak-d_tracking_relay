stateDiagram
    direction TB
    
    [*] --> Init
    Init --> get_stereo_frame

    state "System Loop" as System_Loop {
        direction TB
        state "Stereo Frame abfragen" as get_stereo_frame
        state "Preprocessing" as processing_fork
        state "MediaPipe Augen-Erkennung" as MediaPipe_Face
        state "Landmarks validieren" as Validate_Landmarks
        state "Buffer befüllen" as Add_To_Buffer
        state "MediaPipe Hand-Erkennung" as Run_MediaPipe_Hands
        state "Quadranten zuweisen" as Map_to_Quadranten
        state "Optical Flow" as Optical_Flow_PyrLK
        state "Bewegung validieren" as Validate_Movement
        state "Senden via UDP" as Send_UDP
        state "Frame und Tracking-Daten anzeigen" as Update_UI

        get_stereo_frame --> processing_fork
        
        state processing_fork <<fork>>
        
        state "Augen-Tracker (Haupt Thread)" as EyeTracker {
            direction LR
            
            state "Zustand: SEARCHING (initial)" as Search {
                MediaPipe_Face --> Validate_Landmarks
                Validate_Landmarks --> Add_To_Buffer
                
            }
            
            state "Zustand: TRACKING" as Track {
                Optical_Flow_PyrLK --> Validate_Movement
                
                state "Periodischer Recheck (MediaPipe)" as Recheck
                Validate_Movement --> Recheck : Alle x Frames rechecken
                Recheck --> Validate_Movement : Drift korrigieren
            }
            
            Search --> Track : Buffer beinhaltet genug valide Landmarks
            Track --> Search : Optical Flow fehlgeschlagen oder Drift
        }
        
        state "Han-Tracker (Background Thread)" as HandTracker {
            direction TB
            Run_MediaPipe_Hands --> Map_to_Quadranten
        }
        
        processing_fork --> EyeTracker
        processing_fork --> HandTracker : An background thread senden
        
        state Tracking_Join <<join>>
        
        EyeTracker --> Triangulate
        Triangulate --> Tracking_Join : 3D Koordinaten
        HandTracker --> Tracking_Join : Async Ergebnis
        
        Tracking_Join --> Validate
        Validate --> Send_UDP
        Send_UDP --> Update_UI
        Update_UI --> get_stereo_frame
    }