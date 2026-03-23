stateDiagram
    direction TB
    
    [*] --> Init
    Init --> get_stereo_frame

    state "System Loop" as System_Loop {
        direction TB
        state "Stereo Frame abfragen" as get_stereo_frame
        state "Split" as processing_fork
        state "UI mit Preview und Konfigurations-Slidern" as Update_UI
        state "Tiefeninformationen berechnen" as Triangulate

        get_stereo_frame --> Preprocessing
        Preprocessing --> processing_fork

        state Tracking {
            state processing_fork <<fork>>
            
            state "Augen-Tracker (Haupt Thread)" as EyeTracker {
                direction LR
                Search --> Track
                Track --> Search
            }
            
            processing_fork --> EyeTracker
            processing_fork --> HandTracker

            state Tracking_Join <<join>>
            HandTracker --> Tracking_Join
            EyeTracker --> Triangulate
        }
        Triangulate --> Tracking_Join
        Tracking_Join --> Validate
        Validate --> Relay
        
        state GUI_Condition <<choice>>
        Relay --> GUI_Condition
        
        GUI_Condition --> Update_UI : GUI aktiviert
        Update_UI --> get_stereo_frame : Neue Konfiguration, falls über UI geändert
        GUI_Condition --> get_stereo_frame : GUI deaktiviert (bessere Verarbeitungsrate)
    }
    Update_UI --> Cleanup
    Cleanup --> [*]
    Relay --> Unity