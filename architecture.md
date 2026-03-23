stateDiagram
    direction TB
    
    [*] --> Initialisierung
    Initialisierung --> get_stereo_frame

    state "System Loop" as System_Loop {
        direction TB
        state "Abfrage Stereo Frame" as get_stereo_frame
        state "Vorschau-UI mit Konfigurations-Slidern" as Update_UI
        state "Tiefeninformationen berechnen" as Triangulate
        state "Vorverarbeitung" as Preprocessing
        state "Hand-Tracker (Background Thread)" as HandTracker
        state "Anpassung Kamera Konfiguration" as Update_camera
        state "Validierung der Daten" as Validate
        state "Weitergabe der Daten" as Relay
        
        state processing_fork <<fork>>

        get_stereo_frame --> Preprocessing
        Preprocessing --> processing_fork

        state "Augen-Tracker (Haupt Thread)" as EyeTracker {
            direction LR
            Detect --> Track
            Track --> Detect
        }
        
        processing_fork --> EyeTracker
        processing_fork --> HandTracker

        state Tracking_Join <<join>>
        HandTracker --> Tracking_Join : Hand-Wahrheitswerte
        EyeTracker --> Triangulate
        Triangulate --> Tracking_Join : 3D Iris-Koordinaten
        Tracking_Join --> Validate
        Validate --> Relay
        
        Relay --> Update_UI
    

        state update_Condition <<choice>>
        Update_UI --> update_Condition
        update_Condition --> get_stereo_frame : Keine Konfig-Änderung
        update_Condition --> Update_camera : Konfig-Änderung in UI vorgenommen
        Update_camera --> get_stereo_frame
    }
    Update_UI --> Cleanup : Exit
    Cleanup --> [*]
    Relay --> Unity : UDP