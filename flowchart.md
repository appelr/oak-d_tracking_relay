Flowchart
    direction TB
    
    [*] --> Init
    Init --> get_stereo_frame

    state "System Loop" as System_Loop {
        direction TB
        state "Run Preprocessing" as Preprocessing
        state "Get Stereo Frame" as get_stereo_frame
        state "Processing Fork" as processing_fork
        state "MediaPipe Detect Eyes" as MediaPipe_Face
        state "Validate Landmarks" as Validate_Landmarks
        state "Add to Buffer" as Add_To_Buffer
        state "MediaPipe Detect Hands" as Run_MediaPipe_Hands
        state "Map to Quadrants" as Map_to_Quadranten
        state "Optical Flow" as Optical_Flow_PyrLK
        state "Validate Movement" as Validate_Movement
        state "Send via UDP" as Send_UDP
        state "Display Frame and Tracking Data" as Update_UI


        get_stereo_frame --> Preprocessing
        Preprocessing --> processing_fork
        
        state processing_fork <<fork>>
        
        state "Eye Tracker (Main Thread)" as EyeTracker {
            direction LR
            
            state "SEARCHING (Initial State)" as Search {
                MediaPipe_Face --> Validate_Landmarks
                Validate_Landmarks --> Add_To_Buffer
                
            }
            
            state "TRACKING" as Track {
                Optical_Flow_PyrLK --> Validate_Movement
                
                state "Periodic Recheck (MediaPipe)" as Recheck
                Validate_Movement --> Recheck : Revalidate every x frames
                Recheck --> Validate_Movement : Correct drift
            }
            
            Search --> Track : Buffer contains enough valid landmarks
            Track --> Search : Optical Flow unsuccessful or drift detected
        }
        
        state "Hand Tracker (Background Thread)" as HandTracker {
            direction TB
            Run_MediaPipe_Hands --> Map_to_Quadranten
        }
        
        processing_fork --> EyeTracker
        processing_fork --> HandTracker : Send to background thread
        
        state Tracking_Join <<join>>
        
        EyeTracker --> Triangulate
        Triangulate --> Tracking_Join : 3D Coordinates
        HandTracker --> Tracking_Join : Async Result
        
        Tracking_Join --> Validate
        Validate --> Send_UDP
        Send_UDP --> Update_UI
        Update_UI --> get_stereo_frame
    }