stateDiagram-v2

    state tracker_state <<choice>>

    [*] --> tracker_state : Stereo Frame

    state "Iris-Tracker" as tracker {
        direction TB
        tracker_state --> DETECTION: Tracker State: DETECTION (Initialwert)
        tracker_state --> TRACKING: Tracker State: TRACKING

        state "TrackerState: DETECTION" as DETECTION {
            state "Landmark Detection" as detect
            state validdetection <<choice>>
            state "Detection Buffer leeren" as clearbuffer
            state "Zu Detection Buffer hinzufügen" as addtobuffer
            state bufferfull <<choice>>
            state stabilitycheck <<choice>>
            state "TrackingData setzen" as settrackingdata
            state "Tracker State -> TRACKING" as switchtotracking

            detect --> validdetection
            validdetection --> clearbuffer : Landmarks invalid
            validdetection --> addtobuffer : Landmarks valid
            addtobuffer --> bufferfull
            bufferfull --> stabilitycheck : Detection Buffer voll
            stabilitycheck --> clearbuffer : Stabilitätscheck fehlgeschlagen
            stabilitycheck --> settrackingdata : Stabilitätscheck erfolgreich
            settrackingdata --> switchtotracking

            clearbuffer --> [*]
            bufferfull --> [*] : Detection Buffer nicht voll
            switchtotracking --> [*]
        }

        state "TrackerState: TRACKING" as TRACKING {
            state "Optical Flow" as opticalflow
            state validtracking <<choice>>
            state "Confidence verringern" as decreaseconfidence
            state "TrackingData updaten" as updatetrackingdata
            state plausibletracking <<choice>>
            state recheck <<choice>>
            state "Landmark Detection" as detect2
            state driftdetection <<choice>>
            state "TrackingData updaten" as updatetrackingdata2
            state confidencecheck <<choice>>
            state "TrackerState -> DETECTION" as switchtodetection

            opticalflow --> validtracking
            validtracking --> decreaseconfidence: TrackingData invalid
            validtracking --> plausibletracking : TrackingData valid
            plausibletracking --> decreaseconfidence : TrackingData nicht plausibel
            plausibletracking --> updatetrackingdata : TrackingData plausibel
            decreaseconfidence --> confidencecheck
            confidencecheck --> [*] : Confidence über Minimum
            confidencecheck --> switchtodetection : Confidence unter Minimum
            updatetrackingdata --> recheck
            
            
            switchtodetection --> [*]
            driftdetection --> [*] : Kein OpticalFlow Drift
            updatetrackingdata2 --> [*]
            recheck --> [*] : RecheckInterval nicht erreicht
            recheck --> detect2 : RecheckInterval erreicht

            state "Recheck" as recheckblock{
                detect2 --> driftdetection
                driftdetection --> updatetrackingdata2 : OpticalFlow Drift erkannt
            }
        }
        
    }
    DETECTION --> [*]
    TRACKING --> [*]
    
