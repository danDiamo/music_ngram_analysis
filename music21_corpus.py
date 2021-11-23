"""Exploratory work on melodies in the Ceol Rince na
hEireann corpus, converting each melody from MIDI to feature sequences via the Music21 library;
also finding the root note of each melody via the methods below:

Root detection uses:
1: Music21's implementation of the Krumhansl key-spelling algorithm,
with Krumhansl-Schmucker; Aarden-Essen; Bellman-Budge; and Temperley-Kostka-Payne weightings,
and Craig Sapp's keycor Simple Weights.

2: Key as assigned by the transcriber.

3: Key assigned per Breathnach's method, taking the final note of the piece as the root.

This program extracts and estimates key for every melody in a corpus per the above methods.
It counts the key results obtained for each tune by each method, designates the majority verdict
as the root note of the piece, and returns a simple confidence score indicating the degree of
consensus among detection methods.

'"""

import music21
import os
import pandas as pd
import traceback

from utils import write_to_csv


class Music21Corpus:

    def __init__(self, inpath):
        # MIDI corpus directory:
        self.inpath = inpath
        self.filepaths = os.listdir(self.inpath)
        # List of all melody titles extracted from file names:
        self.melody_titles = [path.split('/')[-1][:-4] for path in self.filepaths if path.endswith('.mid')]
        # List of all melodies in music21 stream format:
        self.melodies = []
        # dict of all tune titles (keys) and melody streams (values)
        self.corpus = {}
        # Pandas DF of key values as assigned and / or detected by methods below:
        self.keys = pd.DataFrame()
        # Pandas DF of root values:
        self.roots = pd.DataFrame()
        # Pandas DF of frequency of occurrence of root values:
        self.root_freq = pd.DataFrame()

    def read_midi_files_to_streams(self):
        for file in self.filepaths:
            if file.endswith('.mid') and not file.startswith('.'):
                mf = music21.midi.MidiFile()
                mf.open((self.inpath + "/" + file).encode())
                try:
                    mf.read()
                    mf.close()
                    s = music21.midi.translate.midiFileToStream(mf, quantizePost=False).flat
                except Exception as exc:
                    print(traceback.format_exc())
                    print(exc)
                self.melodies.append(s)

        return self.melodies

    def combine_melodies_and_titles(self):
        self.corpus = dict(zip(self.melody_titles, self.melodies))
        return self.corpus

    # Q: How does this treat tunes which resolve to chords?
    def reformat_to_numeric_feat_seq(self, output_dir):
        for title, tune in self.corpus.items():
            tune_df = pd.DataFrame(columns=["MIDI_note", "onset", "duration", "velocity"])
            for note in tune.recurse().notes:
                # output and format musical data from music21 stream, and print to dataframe:
                note_df = pd.DataFrame([[
                    round(float(note.offset), 2),
                    round(float(note.duration.quarterLength), 2),
                    note.volume.velocity
                    ]],
                    columns=["onset", "duration", "velocity"])

                if note.isNote:
                    note_df["MIDI_note"] = int(note.pitch.ps)

                if note.isChord:
                    tune_df["MIDI_note"] = int(note.root().ps)

                tune_df = tune_df.append(note_df, ignore_index=True)

            # write output file
            tune_df.to_csv((output_dir + "/" + title + ".csv"))

    def run_key_detection_algs(self):

        self.keys['title'] = self.corpus
        self.keys.reset_index(inplace=True, drop=True)

        krumhansl = [tune.analyze('key') for tune in self.corpus.values()]
        self.keys['Krumhansl-Shmuckler'] = krumhansl

        simple = [music21.analysis.discrete.SimpleWeights(tune).getSolution(tune)
                  for tune in self.corpus.values()]
        self.keys['simple weights'] = simple

        aarden = [music21.analysis.discrete.AardenEssen(tune).getSolution(tune)
                  for tune in self.corpus.values()]
        self.keys['Aarden Essen'] = aarden

        bellman = [music21.analysis.discrete.BellmanBudge(tune).getSolution(tune)
                  for tune in self.corpus.values()]
        self.keys['Bellman Budge'] = bellman

        temperley = [music21.analysis.discrete.TemperleyKostkaPayne(tune).getSolution(tune)
                  for tune in self.corpus.values()]
        self.keys['Temperly Kostka Payne'] = temperley

        return self.keys

    def extract_transcriber_assigned_key(self):
        self.keys['as transcribed'] = [tune[1] for tune in self.corpus.values()]
        return self.keys

    # this method no longer necessary?
    def reformat_music21_objs_to_str(self):
        self.keys = self.keys.astype('string')
        return self.keys

    # this also method no longer necessary?
    def find_most_frequent_key(self):
        self.keys['key'] = self.keys.mode(axis=1, numeric_only=False)[0]
        return self.keys

    def extract_roots_from_keys(self):
        self.roots = self.keys.copy(deep=True)
        for res in ['Krumhansl-Shmuckler',
                    'simple weights',
                    'Aarden Essen',
                    'Bellman Budge',
                    'Temperly Kostka Payne',
                    'as transcribed']:
            self.roots[res] = self.roots[res].apply(lambda x: x.tonic.name.upper())
            # this uses music21's note.Note.name to access pitch name (note letter, and sharp/flat if it has one)

        return self.roots

    def extract_final_note(self):
        final_notes = []
        for tune in self.corpus.values():
            last_note = tune.recurse().notes[-1]
            print(last_note)
            if last_note.isNote:
                final_notes.append(last_note.name.upper())
            elif last_note.isChord:
                final_notes.append(last_note.root().name.upper())
            else:
                final_notes.append('')

        self.roots['final_note'] = final_notes
        return self.roots

    # this has been refactored and moved to MusicDataCorpus
    def count_freq_of_root_occurrences(self):
        # Counts occurrences of each root note name in each row of dataframe.
        roots = self.roots.copy(deep=True)
        # Temporarily remove the 'title' column so that it is not included in the count:
        roots.pop('title')
        self.root_freq = roots.apply(pd.Series.value_counts, axis=1, normalize=True)
        certainty_vals = self.root_freq.max(axis=1).to_numpy()
        certainty_vals = certainty_vals.round(decimals=2)
        self.roots['certainty'] = certainty_vals
        self.root_freq.insert(0, 'title', self.roots['title'])
        return self.root_freq

    # this has been moved to MusicDataCorpus
    def find_most_frequent_root(self):
        self.roots['root'] = self.roots.mode(axis=1, numeric_only=False)[0]
        return self.roots


corpus = Music21Corpus("/Users/dannydiamond/NUIG/Polifonia/CRE/CRE_MIDI")
print(corpus.filepaths)
print('\n\n')
print(corpus.melody_titles)
corpus.read_midi_files_to_streams()
corpus.combine_melodies_and_titles()
for i, j in corpus.corpus.items():
    print(i, [note for note in j])
corpus.run_key_detection_algs()
corpus.extract_transcriber_assigned_key()
corpus.extract_roots_from_keys()
print(corpus.roots.head())
corpus.extract_final_note()

corpus.reformat_music21_objs_to_str()
print(corpus.keys.head())
corpus.find_most_frequent_key()
print(corpus.keys.head())
corpus.count_freq_of_root_occurrences()
corpus.find_most_frequent_root()
print(corpus.root_freq.head())

roots_outpath = '/Users/dannydiamond/NUIG/Polifonia/CRE/root_detection'
write_to_csv(corpus.roots, roots_outpath, 'music21_roots')
write_to_csv(corpus.root_freq, roots_outpath, 'music21_freqs')
write_to_csv(corpus.keys, roots_outpath, 'music21_keys')

numeric_corpus_outpath = '/Users/dannydiamond/NUIG/Polifonia/CRE/CRE_primary_feat_seq'
corpus.reformat_to_numeric_feat_seq(numeric_corpus_outpath)

