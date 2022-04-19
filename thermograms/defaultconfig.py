# -*- coding: utf-8 -*-
"""
Default-Konfiguration für die Auswertung der thermografischen Daten

Author: Andreas Pestel
"""

config = {# Pfadangaben
          'PATH': 'D:\Thermal evaluation\Thermal_data',  # Pfad der Versuche
          'ConfigFileList' : 'evaluation-list.conf',    # Name der Config-File, die die Liste der auszuwertenden Versuche enthält
          'ConfigFileName' : 'evaluation-config.conf',  # Name der versuchsspezifischen Config-File
          
          # Versuchsdaten
          'Versuchsbezeichnung' : None,                 # Bezeichnung des Versuchs
          'Beschreibung' : None,                        # Beschreibung des Versuchs
          'evaluationArea' :[],                         # Auswertebereich einschränken, Angabe in Pixel [links, rechts, oben, unten]
          'temperatureTimeCurves' : [],                 # Temperatur-Zeit-Kurve für einzelne Pixel erstellen, Format 'P1 : 182/57' (Spalte/Zeile), bei mehreren Kurven durch Komma trennen

          # Allgemein
          'supportedCameras' : ['VarioTHERM',
                                'Optris'],              # unterstützte Kameramodelle
          'evaluationPath' : 'Evaluation',              # Dateipfad zur Auswertung
          'initialPhase' : 1,                           # Anlaufzeit des Versuchs in Sekunden
          'temperaturDelta' : 1.0,                      # Temperaturschwelle für Übergängen in den Auswertephasen
          'timeForRadiationPhase' : 0.5,                # Betrachtungsdauer in s für die Eigenstrahlungsphase
          'IgnoreTimeAtStart' : 0.0,                    # Zeitspanne in s, welche zum Beginn der Messung ignoriert wird
          'plot3DElevation' : 65,                       # Anhebung des Plots der Temperaturprofile
          'plot3DAzimuth' : None,                       # Drehung des Plots der Temperaturprofile
          'plot3DXLabel' : 'Breite [Pixel]',            # Beschriftung X-Achse im 3D-Plot
          'plot3DYLabel' : 'Höhe [Pixel]',              # Beschriftung Y-Achse im 3D-Plot
          'plot3DZLabelIntegral' : '∫ ∆T dt [K/s]',     # Beschriftung Z-Achse im 3D-Plot für Integral
          'plot3DZLabelRise' : 'm [K/s]',               # Beschriftung Z-Achse im 3D-Plot für Anstiege
          'plot3DWidth' : 16.0,                         # Breite des 3D Plots in cm
          'plot3DHeight' : 12.0,                        # Höhe des 3D Plots in cm
          'plot3DDPI' : 300,                            # DPI des Plots
          'plot3DFileFormat' : 'png',                   # Dateiformat der 3D-Plots
          'plot2DWidth' : 16.0,                         # Breite des 3D Plots in cm
          'plot2DHeight' : 12.0,                        # Höhe des 3D Plots in cm
          'plot2DDPI' : 300,                            # DPI des Plots
          'plot2DFileFormat' : 'png'                    # Dateiformat der 2D-Plots
          }

# Konfiguration für einzelne IR-Kamera-Modelle
cameraModel = {'VarioTHERM' : {'camera' : 'VarioTHERM',                  # Kameramodell
                               'dataPath' : 'VarioTHERM',                # Dateipfad
                               'decimalSign' : '.',                      # Zeichen für Dezimaltrenner
                               'delimiter' : ';',                        # Zeichen für Spaltentrenner
                               'linesToSkip' : 1,                        # Kopfzeilen, welche vor Auswertung entfernt werden
                               'lines' : 256,                            # Zeilen zur Auswertung
                               'columns' : 256,                          # Spalten zur Auswertung
                               'changeTemperatureToCelsius' : True,      # Umrechnung von K zu °C
                               'frequency' : 50                          # Aufnahmefreuqnz der Kamera in Herz
                               },
               'Optris' : {'camera' : 'Optris',                          # Kameramodell
                           'dataPath' : 'Optris',                        # Dateipfad
                           'decimalSign' : ',',                          # Zeichen für Dezimaltrenner
                           'delimiter' : ';',                            # Zeichen für Spaltentrenner
                           'linesToSkip' : 0,                            # Kopfzeilen, welche vor Auswertung entfernt werden
                           'lines' : 480,                                # Zeilen zur Auswertung
                           'columns' : 640,                              # Spalten zur Auswertung
                           'changeTemperatureToCelsius' : False,         # Umrechnung von K zu °C
                           'frequency' : 32                              # Aufnahmefreuqnz der Kamera in Herz
                          }
               }
