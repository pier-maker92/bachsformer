import numpy as np
from scipy import stats
from scipy.signal import gaussian as gaussian
from mido import MidiFile,tempo2bpm,tick2second
import mido

class Midi():
    def __init__(self):
        note_map = {0: 'A', 1: 'Bb', 2: 'B', 3: 'C',
                    4: 'C#', 5: 'D', 6: 'Eb', 7: 'E',
                    8: 'F', 9: 'F#', 10: 'G', 11: 'Ab'
        }
        midi_map = dict()
        for i in range(88):
            midi_map[i+1+20] = f"{note_map[i%12]}_{max(0,i-3)//12}"
        midi_map[0] = 'pause'
        self.midi_map = midi_map
        self.note_map = {v: k for k, v in midi_map.items()}
        
class MidiMiniature(Midi):
    """Convert a midi file into a discrete representation (0,1) with 88 rows
    and x columns, where x depends on the level of desired granularity
    x = n_quaters * miniature_quarter_ticks"""
    def __init__(self,n_quaters,midi_quarter_ticks=480,miniature_quarter_ticks=32):
        super().__init__()
        self.n_quaters = n_quaters
        self.midi_quarter_ticks = midi_quarter_ticks
        self.bar_length =self.midi_quarter_ticks*self.n_quaters
        self.miniature_ticks=miniature_quarter_ticks*self.n_quaters
        self.compress_ratio = self.bar_length/self.miniature_ticks
        self.empty_bar_miniature = np.zeros([88,self.miniature_ticks])

    def preprocess_midi(self,filepath):
        tracks = MidiFile(filepath).tracks
        events = []
        for track in tracks:
            cur_time = 0; 
            notes_on = np.empty(0).astype(int); start_events = np.empty(0).astype(int)
            for msg in track:
                if msg.is_meta:continue
                elif msg.type.split("_")[0] == "note":
                    note = msg.note - 21 # 0 means A0
                    cur_time += msg.time
                    # note on
                    if msg.velocity and msg.type=="note_on":
                        # update notes one
                        notes_on = np.hstack( (notes_on,note) )
                        start_events = np.hstack( (start_events,cur_time) )
                    # note off
                    else:
                        note_idx = np.where(notes_on==note)[0]
                        start_event = start_events[note_idx][0]
                        events.append(((start_event,cur_time,note)))
                        notes_on = np.delete(notes_on,note_idx); start_events = np.delete(start_events,note_idx)
        events.sort()
        return events

    def refine_miniature(self,bars):
        n_bars = len(bars)
        bars = np.concatenate(bars,axis=1)
        for i in range(88):
            note_on_indx = np.where(bars[i]==2)[0] 
            note_on_indx_true = note_on_indx[np.where(note_on_indx>0)[0]]
            bars[i][note_on_indx_true - 1] = 0
            bars[i][note_on_indx] = 1
        return [bars[:,np.arange(self.miniature_ticks)+i*self.miniature_ticks] for i in range(n_bars)]

    def make_miniature(self,filepath):
        events = self.preprocess_midi(filepath)
        bars = []; tails_to_add = []
        bar_miniature = self.empty_bar_miniature.copy()
        cur_bar=0
        for start, end, note in events:
            start_bar_index = start//self.bar_length
            if start_bar_index>cur_bar:
                for i in range(start_bar_index-cur_bar):
                    bars.append(bar_miniature)
                    bar_miniature = self.empty_bar_miniature.copy()
                    if len(tails_to_add):
                        #print(f"tails_to_add : {tails_to_add}")
                        t_start,t_end,t_note = tails_to_add[0]
                        grid_start_index = round((t_start%self.bar_length)/self.compress_ratio)
                        gird_end_index = round((t_end%self.bar_length)/self.compress_ratio)
                        bar_miniature[t_note,grid_start_index:gird_end_index] = 1
                        tails_to_add.pop(0)
                cur_bar = start_bar_index
            # start miniature grid index
            grid_start_index = round((start%self.bar_length)/self.compress_ratio)
            # check if note fall entirely in currnt bar, otherwise store the tail and add it later
            end_bar_index = end//self.bar_length
            if start_bar_index<end_bar_index: 
                gird_end_index = self.miniature_ticks
                for i in range(1,end_bar_index-start_bar_index):
                    next_start = (start_bar_index+i)*self.bar_length
                    next_end = (start_bar_index+i+1)*self.bar_length
                    tails_to_add.append((next_start,next_end,note))
                if not end_bar_index-start_bar_index-1: next_end = (end_bar_index)*self.bar_length
                if end%self.bar_length: tails_to_add.append((next_end,end,note))
            else: gird_end_index = round((end%self.bar_length)/self.compress_ratio)
            bar_miniature[note,grid_start_index:gird_end_index] = 1
            bar_miniature[note,grid_start_index] = 2 # note on event
        bars.append(bar_miniature)
        return self.refine_miniature(bars)

    def miniature2midi(self,bars):
        # setup new midifile
        midi_file = MidiFile(ticks_per_beat=self.midi_quarter_ticks)
        track = mido.MidiTrack()
        time_signature = mido.MetaMessage('time_signature', numerator=self.n_quaters, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0)
        key_signature = mido.MetaMessage('key_signature', key='G', time=0)
        for item in [time_signature,key_signature]: 
            track.append(item)
        midi_file.tracks.append(track)
        track = mido.MidiTrack()
        # write messages
        n_bars = len(bars)
        bars = np.concatenate(bars,axis=1)
        events = []; prev = 0
        for note_idx,note in enumerate(bars):
            note = np.insert(note,0,0)
            note_diff = np.diff(note)
            # notes on
            notes_on = np.where(note_diff==1)[0]
            for grid_idx in notes_on: 
                grid_idx = round(grid_idx*self.compress_ratio)
                events.append( (grid_idx,'note_on',note_idx) )
            # notes off
            notes_off = np.where(note_diff==-1)[0]
            for grid_idx in notes_off: 
                grid_idx = round(grid_idx*self.compress_ratio)
                events.append( (grid_idx,'note_off',note_idx) )
        events.sort()
        for e in events:
            t = e[0] - prev; prev = e[0]
            track.append(mido.Message(e[1],note=e[2]+21, velocity=100, time=t))
            
        midi_file.tracks.append(track)
        return midi_file