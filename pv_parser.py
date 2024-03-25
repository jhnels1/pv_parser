import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import spectrogram, periodogram, find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
from shapely.geometry import Polygon

def parse_cycles( PV_data, fs ):
    dt = 1./fs
    dPdt = gaussian_filter1d( PV_data["PV Pressure"], 5, order=1 )*fs
    peaks, info = find_peaks( -dPdt, prominence=100, distance=10 )

    P_cycles = np.split( PV_data["PV Pressure"], peaks )
    V_cycles = np.split( PV_data["PV Volume"], peaks )

    
    tgrid = np.arange(len(dPdt))*dt
    t_cycles = np.split( tgrid, peaks )
    # We only need the start time for each cycle
    t_start = np.array( [ cycle[0] for cycle in t_cycles ] )
    
    return V_cycles, P_cycles, t_start

class PV_cycle:
    def __init__(self, V_data, P_data, t0, fs):
        self.fs = fs
        self.dt = 1./fs
        self.t0 = t0
        self.V_data = V_data
        self.P_data = P_data
        self.prep_data()
        self.find_ES_ED()

    def check_closed(self):
        # Make sure the loop is closed
        return (self.V_data.iloc[-1]==self.V_data.iloc[0])&(self.P_data.iloc[-1]==self.P_data.iloc[0])
    
    def prep_data(self):

        # Close the loop
        if not self.check_closed():
            self.V_data = np.append( self.V_data, self.V_data.iloc[0] )
            self.P_data = np.append( self.P_data, self.P_data.iloc[0] )        
        
        self.Pmax = np.amax( self.P_data )
        self.Pmin = np.amin( self.P_data )
        self.Vmax = np.amax( self.V_data )
        self.Vmin = np.amin( self.V_data )
        
        self.V0 = np.mean( self.V_data )
        self.P0 = np.mean( self.P_data )
        self.x = ( self.V_data - self.V0 )/(self.Vmax-self.Vmin)
        self.y = ( self.P_data - self.P0 )/(self.Pmax-self.Pmin)
        self.r = np.sqrt( self.x**2 + self.y**2 )
        self.theta = np.arctan2( self.y, self.x )
        self.t = np.arange(len(self.V_data))/self.fs
        self.tlen = np.amax(self.t)

        # Perimiter and area of scaled, dimensionless loop
        pgon_x = Polygon( zip(self.x, self.y) )
        self.perim_x = pgon_x.length 
        self.area_x = pgon_x.area

        # Perimiter and area of physical loop
        pgon_p = Polygon( zip(self.V_data, self.P_data) )
        self.perim_p = pgon_p.length # Units are nonsense
        self.area_p = pgon_p.area # Stroke work, (mmHg)*(mL)

        self.xmax = np.amax( self.x )
        self.xmin = np.amin( self.x )
        self.ymax = np.amax( self.y )
        self.ymin = np.amin( self.y )



    def find_ES_ED(self):
        # Experimental feature- not guarenteed to work
        
        # Filter normalized traces and calculate derivatives
        xp = savgol_filter( self.x, 15, 3, deriv=1, delta=self.dt, mode='wrap' )
        xpp = savgol_filter( self.x, 15, 3, deriv=2, delta=self.dt, mode='wrap' )

        yp = savgol_filter( self.y, 15, 3, deriv=1, delta=self.dt, mode='wrap' )
        ypp = savgol_filter( self.y, 15, 3, deriv=2, delta=self.dt, mode='wrap' )
        
        # Calculate curvature
        k = (xp*ypp-yp*xpp)/np.power(xp**2+yp**2, 1.5)

        # Save some relevant curvature paramaters
        self.k_data = k
        self.kmax = np.amax(np.abs(k))
        self.kmean = np.mean( np.abs(k) )

        # restrict our search periods for ES and ED
        in_ES = (self.x<0)&(self.y>0)
        in_ED = (self.x>0)&(self.y<0)

        mag = self.x**2 + self.y**2
        
        k_ES = np.copy(k)
        k_ES[~in_ES] = 0
        k_ED = np.copy(k)
        k_ED[~in_ED] = 0  

        mag_ES = np.copy(mag)
        mag_ES[~in_ES] = 0
        mag_ED = np.copy(mag)
        mag_ED[~in_ED] = 0

        peak_ES = np.argmax( np.abs(k_ES) )
        peak_ED = np.argmax( np.abs(k_ED) )
        #peak_ES = np.argmax( mag_ES )
        #peak_ED = np.argmax( mag_ED )
        
        #print( len(peaks_ES), len(peaks_ED) )
        
        self.P_ED = self.P_data[ peak_ED ]
        self.V_ED = self.V_data[ peak_ED ]

        self.P_ES = self.P_data[ peak_ES ]
        self.V_ES = self.V_data[ peak_ES ]

    def plot_loop(self):

        V_smooth = gaussian_filter1d( self.V_data, 2, order=0 )
        P_smooth = gaussian_filter1d( self.P_data, 2, order=0 )
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot( self.V_data, self.P_data, label='Raw Data' )
        #ax.plot( V_smooth, P_smooth, alpha=0.5, label='Filtered Data' )

        ax.scatter( self.V_ED, self.P_ED, c='r', marker='o', label='ED', s=100 )
        ax.scatter( self.V_ES, self.P_ES, c='r', marker='x', label='ES', s=100 )
        plt.xlabel("LV Volume [mL]")
        plt.ylabel("LV Pressure [mmHg]")
        plt.xlim(0, 60)
        plt.ylim(0, 130)
        plt.grid()
        plt.legend()
        plt.show()

    def get_features(self):
        feature_dict = {
            "t0":self.t0,
            "tlen":self.tlen,
            "Vmean":self.V0,
            "Pmean":self.P0,
            "Vmax":self.Vmax,
            "Vmin":self.Vmin,
            "Pmax":self.Pmax,
            "Pmin":self.Pmin,
            "Work":self.area_p,
        }

        return feature_dict

def loop_table( data, fs ):
    V_cycles, P_cycles, t0 = parse_cycles( data, fs )

    ret_table = list()
    for i in range(len(t0)):
        my_cycle = PV_cycle( V_cycles[i], P_cycles[i], t0[i], fs )
        cycle_table = my_cycle.get_features()
        ret_table.append( cycle_table )

    return pd.DataFrame.from_dict(ret_table), V_cycles, P_cycles, t0

if __name__=="__main__":
    main()