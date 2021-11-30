from matplotlib import pyplot as plt, cm, ticker
import numpy as np

import setup_corpus.constants as constants
from utils import *



class MusicData:

    def __init__(self, fin):
        music_data = read_csv(fin)
        self.title = music_data[0]  # melody title
        self.music_data = music_data[1]  # feature sequence dataframe
        # feature sequence dataframe of accented notes only:
        self.music_data_accents = filter_dataframe(self.music_data, seq='velocity')
        self.weighted_music_data = None  # will hold duration-weighted feature sequence dataframe
        # will hold duration-weighted feature sequence dataframe of accented notes only:
        self.weighted_music_data_accents = None
        self.most_freq_note = None  # will hold most frequently-occurring MIDI note value from self.music_data
        # will hold most frequently-occurring accented MIDI note value from self.music_data_accents:
        self.most_freq_acc = None
        self.most_freq_notes = None

    def add_dur_eighth_notes(self):
        self.music_data['duration'] = self.music_data['duration'] * 2
        print(self.music_data.head())
        return self.music_data

    def recalc_onset_eighth_notes(self):
        self.music_data['onset'] = self.music_data['onset'] * 2
        print(self.music_data.head())
        return self.music_data

    def extract_feat_seq(self, col_name):
        # returns selected column as numpy array
        return extract_feature_sequence(self.music_data, col_name)

    def extract_accents_feat_seq(self, col_name):
        # returns selected column as numpy array
        return extract_feature_sequence(self.music_data_accents, col_name)

    def find_most_frequent_val_in_feat_seq(self, col_name):
        find_most_frequent_value_in_seq(self.music_data, col_name)

    def find_most_frequent_val_in_accents_feat_seq(self, col_name):
        find_most_frequent_value_in_seq(self.music_data_accents, col_name)

    def generate_duration_weighted_music_data(self, feat_seqs):
        if self.weighted_music_data is None:
            self.weighted_music_data = pd.DataFrame()

        for seq in feat_seqs:
            onsets = self.music_data['onset'].to_numpy()
            seq_recalc = self.music_data[seq].to_numpy()

            eighths = np.arange(int(onsets[0]), int(onsets[-1]) + 1)
            idx = np.searchsorted(onsets, eighths)

            dur_weighted_seq = [seq_recalc[i] for i in idx]
            self.weighted_music_data[f'dur weighted {seq}'] = dur_weighted_seq

        self.weighted_music_data = self.weighted_music_data.rename_axis('note event')
        print('\n' + self.title + ' weighted note sequence:')
        print(self.weighted_music_data.head())
        return self.weighted_music_data

    def filter_duration_weighted_music_data_accents(self):
        # filter duration-weighted dataframe to keep only accented notes
        self.weighted_music_data_accents = filter_dataframe(self.weighted_music_data, seq='dur weighted velocity')
        print('\n' + self.title + ' weighted accents sequence:')
        print(self.weighted_music_data_accents.head(10))
        return self.weighted_music_data_accents

    def find_most_freq_note(self):
        # find most common MIDI note number in duration-weighted pitch sequence:

        most_freq_note = find_most_frequent_value_in_seq(self.music_data, seq='MIDI_note')
        # most_freq_acc = find_most_frequent_value_in_seq(self.music_data_accents, seq='MIDI_note')
        most_freq_weighted_note = find_most_frequent_value_in_seq(self.weighted_music_data,
                                                                  seq='dur weighted MIDI_note')
        # most_freq_weighted_acc = find_most_frequent_value_in_seq(self.weighted_music_data_accents,
        #                                                            seq='dur weighted MIDI_note')

        res = [most_freq_note, most_freq_weighted_note]
        labels = ['freq note', 'freq weighted note']

        if self.most_freq_notes is None:
            self.most_freq_notes = dict(zip(labels, res))


    def find_most_freq_acc_note(self):

        if self.most_freq_acc is None:
            self.most_freq_acc = find_most_frequent_value_in_seq(self.weighted_music_data_accents,
                                                                   seq='dur weighted MIDI_note')
        return self.most_freq_acc


class MusicDataCorpus:

    def __init__(self, indir):
        files = [indir + '/' + name for name in os.listdir(indir) if not name.startswith('.')]
        self.corpus = [MusicData(file) for file in files]
        self.titles = [tune.title for tune in self.corpus]
        self.music_data_roots = None  # will hold dataframe of MusicData-derived most frequent MIDI notes.
        self.normalized_roots = None
        self.root_names = None
        self.music21_roots = None  # will hold dataframe of Music21 root detection results.
        self.roots = None  # merged roots dataframe, basis for root detection calculations
        self.root_freqs = None
        self.roots_numeric = None

    def recalc_duration(self):
        for tune in self.corpus:
            tune.add_dur_eighth_notes()
        return self.corpus

    def recalc_onsets(self):
        for tune in self.corpus:
            tune.recalc_onset_eighth_notes()
        return self.corpus

    def calculate_duration_weighted_feat_seq_data(self, feat_seq):
        for tune in self.corpus:
            tune.generate_duration_weighted_music_data(feat_seq)
        return self.corpus

    def filter_duration_weighted_accents(self):
        for tune in self.corpus:
            tune.filter_duration_weighted_music_data_accents()
        return self.corpus

    def find_most_freq_notes_in_pitch_seq(self):
        for tune in self.corpus:
            tune.find_most_freq_note()
            print(f'For {tune.title}:')
            print(f'Most frequently-occurring values in MIDI note sequences: {tune.most_freq_notes}')
        return self.corpus

    def populate_musicdata_roots_df(self):
        # Generate a table containing the most frequent MIDI note numbers in weighted and unweighted
        # note- and accent- series for every MusicData object in the corpus.
        titles = [tune.title for tune in self.corpus]
        roots = [tune.most_freq_notes for tune in self.corpus]
        if self.music_data_roots is None:
            self.music_data_roots = pd.DataFrame(roots)
        self.music_data_roots.insert(0, 'title', titles)
        self.music_data_roots.sort_values(by='title', inplace=True)
        print(self.music_data_roots.head())
        print(f'Checksum: MusicData MIDI root values: {len(self.music_data_roots)}')
        return self.music_data_roots

    def noramalize_root_values(self):
        # convert numeric root values from MIDI note numbers to values from 0-11, where 0=C and 11=B
        # (I.E.: chromatic pitch classes with C as root)
        normalized_roots = self.music_data_roots.copy()
        col_names = ['freq note', 'freq weighted note']
        for name in col_names:
            normalized_roots[name] = self.music_data_roots[name] % 12

        if self.normalized_roots is None:
            self.normalized_roots = normalized_roots
            print(self.normalized_roots.head())
            print(f'Checksum: Normalized MusicData MIDI root values: {len(self.normalized_roots)}')

        return self.normalized_roots

    def convert_normalized_roots_to_note_names(self):
        # NOTE: below could possibly be implemented via DataFrame.merge():
        lookup = dict(zip(constants.lookup_table['root num'], constants.lookup_table['note names']))
        root_names = self.normalized_roots.copy()
        root_names.set_index('title', inplace=True)
        # lookup vals for entire dataframe:
        root_names = root_names.applymap(lambda x: lookup[x])
        if self.root_names is None:
            self.root_names = root_names
        print(self.root_names.head())
        print(f'Checksum: MusicData root name values: {len(self.root_names)}')

        return self.root_names

    def convert_root_names_to_root_nums(self):
        # converts root note names to root number (pitch class relative to C)
        # appends result to self.roots dataframe
        roots = self.roots.copy()
        # lookup numeric root vals & map to new column:
        lookup = dict(zip(constants.lookup_table['note name'], constants.lookup_table['pitch class']))
        roots = roots.replace(lookup)
        roots.set_index('title', inplace=True)
        self.roots_numeric = roots
        print(roots.dtypes)
        print(self.roots_numeric.head())
        self.roots_numeric.apply(pd.to_numeric, errors='coerce')
        print(self.roots_numeric.dtypes)
        print(f'Checksum: {len(self.roots_numeric)}')
        return self.roots_numeric

    def convert_root_nums_to_midi_note_nums(self):
        # converts root number (pitch class relative to C) to MIDI note number for 4th octave.
        # appends result to self.roots dataframe
        roots = self.roots.copy()
        roots['MIDI_root'] = roots['root num'].map(constants.lookup_table.set_index('root num')['midi num'])
        self.roots = roots
        print(self.roots.head())
        return self.roots

    def read_music21_roots(self, music21_path):
        music21_roots = read_csv(music21_path)[1]
        if self.music21_roots is None:
            self.music21_roots = music21_roots
            print(self.music21_roots.head())
            print(f'Checksum: Music21 root values: {len(self.music21_roots)}')
        return self.music21_roots

    def concatenate_music21_and_music_data_roots(self):

        if self.roots is None:
            self.roots = pd.merge(self.music21_roots, self.root_names, on='title', how='left')
            print(self.roots.head())
            print(f'Checksum: combined root values: {len(self.roots)}')
        return self.roots

    def calculate_root_result_confidence_scores(self):
        # Counts occurrences of each root note name in results for each tune. Returns number of occurrences
        # of the most frequent root value / total number of results, giving a simple confidence score for root
        # assignment.

        roots_table = self.roots.copy(deep=True)
        # Temporarily remove the 'title' column so that it is not included in the count:
        titles = roots_table.pop('title')
        root_freq = roots_table.apply(pd.Series.value_counts, axis=1, normalize=True)
        certainty_vals = root_freq.max(axis=1).to_numpy()
        certainty_vals = certainty_vals.round(decimals=2)
        roots_table['certainty'] = certainty_vals
        roots_table.insert(0, 'title', titles)
        self.roots = roots_table
        print('Roots dataframe with confidence score:')
        print(self.roots.head())
        return self.roots

    def append_expert_assigned_root_data(self, inpath):
        # extracts expert-assigned root values from given csv table, and appends to self.roots dataframe
        expert_assigned_roots_table = read_csv(inpath, dtype=str)
        expert_assigned_roots_dataframe = expert_assigned_roots_table[1]
        print(expert_assigned_roots_dataframe.head())
        expert_assigned_roots = expert_assigned_roots_dataframe.pop('root')
        print(expert_assigned_roots.head())
        print(f'Checksum: number of expert-assigned values: {len(expert_assigned_roots)}')
        print(f"Checksum: number of automatically-detected values: {len(self.roots)}")
        # merge
        self.roots['expert assigned'] = expert_assigned_roots
        print(self.roots.head())
        return self.roots

    def calculate_percentage_agreement_between_metrics_and_expert_assigned_vals(self, outpath):
        # create boolean helper column to calculate % agreement between 'mode' and 'expert assigned' values
        # using above, calculate % agreement between each metric and expert-assigned root.
        # print and save report
        roots = self.roots_numeric.copy()
        metrics = roots.columns[1:-1]
        percentages = []
        report = pd.DataFrame({'metric': metrics})
        for metric in metrics:
            roots[f'{metric} / expert agreement'] = np.where(
                roots[metric] == roots['expert assigned'], 1, 0)
            percentage_agreement = roots[f'{metric} / expert agreement'].sum() / roots.shape[0]
            percentage_agreement = round(percentage_agreement, 2)*100
            percentages.append(percentage_agreement)
            print(f'\nPercentage agreement between {metric} and expert-assigned root:')
            print(f'{percentage_agreement}%\n')
        report['% agreement'] = percentages
        write_to_csv(report, outpath, 'percentage_agreement_with_expert_assigned_roots')

        return self.roots_numeric

    def preprocess_root_freq_data(self):
        # generates self.root_freqs dataframe, for input into Fleiss' kappa calculations
        roots = self.roots.copy()
        # Temporarily remove the 'title' column so that it is not included in the count:
        titles = roots.pop('title')
        if self.root_freqs is None:
            self.root_freqs = roots.apply(pd.Series.value_counts, axis=1, normalize=False)
        self.root_freqs.fillna(0, inplace=True)
        float_cols = self.root_freqs.select_dtypes(include=['float64'])
        for col in float_cols.columns.values:
            self.root_freqs[col] = self.root_freqs[col].astype('int64')
        self.root_freqs.insert(0, 'title', titles)
        self.root_freqs.set_index('title', inplace=True)
        print("\n\nFleiss Kappa: input data\n")
        print(self.root_freqs.head())
        return self.root_freqs

    # def calculate_krippendorfs_alpha(self):
    #     roots = self.roots.copy()
    #     roots.set_index('title', inplace=True)
    #     root_results = roots.pop('root')
    #
    #     for col in roots:
    #         new_df = pd.DataFrame()
    #         new_df[col] = col
    #         new_df['root'] = root_results
    #         kd_alpha = krippendorff.alpha(new_df)


    # def calculate_fleiss_kappa(self):
    #     data = self.root_freqs.values
    #     res = fleiss_kappa(data, method='fleiss')
    #     print(res)
    #     return res

    def calculate_correlation_matrix(self, filename: str, plot_title: str):

        self.roots_numeric.pop('certainty')
        res = np.triu(self.roots_numeric.corr().values, 1)
        res_df = pd.DataFrame(res, columns=self.roots_numeric.columns)
        res_df['name'] = self.roots_numeric.columns
        res_df.set_index('name', inplace=True)
        print(f"\n\nRoot detection:\n{plot_title}\n:")
        print(res_df)
        filepath = '/Users/dannydiamond/NUIG/Polifonia/CRE/root_detection/aggregated_results/'
        data_outfile = f'{filename}_data'
        write_to_csv(res_df, filepath, data_outfile)

        # labels = res_df.columns.tolist()
        # print([l for l in labels])
        # labels_lst = [alg_name.split(' ') for alg_name in labels]
        # initials = [[f'{word[0].upper()}. ' for word in lst] for lst in labels_lst]
        # alg_initials = [(''.join(i)).rstrip() for i in initials]
        # print(alg_initials)

        labels = [num for num in '123456789'] + ['Mode', 'EA']
        alg_labels = [alg for alg in self.roots.columns[1:-3]]
        annotation = [f'{i[0] + 1}: {i[1]}' for i in enumerate(alg_labels)]
        annotation_formatted = '\n'.join(annotation) + '\nMode: Mode of results\nEA: Expert-assigned root'

        print('\n\nCorrelation plot:\n')
        fig = plt.figure()
        fig.set_size_inches(12, 8, forward=True)
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', direction='out')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        fig.suptitle(f'Root detection: {plot_title}')
        cmap = cm.get_cmap('gray_r', 10)
        cax = ax.matshow(res, interpolation="nearest", cmap=cmap)
        loc = ticker.MultipleLocator(base=1.0)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        fig.colorbar(cax)
        plt.figtext(0.035, 0.6, annotation_formatted, fontsize=11)
        plt.subplots_adjust(left=0.25)
        plt.autoscale(tight=True)
        plt.tight_layout()
        fig_outfile = f'{filename}_fig.pdf'
        plt.savefig(filepath + fig_outfile, dpi=360, bbox_inches="tight")
        plt.show()

        return res

    def find_most_frequent_root_results(self):
        # Returns mode of values for each row in dataframe. I.E.: the most frequently occurring root value
        # for each tune in the corpus.
        self.roots['root'] = self.roots.mode(axis=1, numeric_only=False)[0]
        return self.roots

    # seems unnecessary?
    def write_roots_dataframe_to_csv(self, path, filename):
        write_to_csv(self.roots, path, filename)
        return f'Root note results saved to:\n{path}/{filename}'

    def calc_pitch_class_values(self):
        # calculate key-invariant pitch class value for every note in every tune in corpus
        self.roots.reset_index(inplace=True)
        print(self.roots.head())
        for tune in self.corpus:
            midi_root = self.roots[self.roots['title'] == tune.title]['MIDI_root']
            midi_root = int(midi_root)
            tune.music_data['MIDI_root'] = midi_root
            tune.music_data['key_invar_pitch'] = tune.music_data['MIDI_note'] - midi_root
            tune.music_data['chromatic_pitch_class'] = tune.music_data['key_invar_pitch'] % 12
            print(tune.music_data.head())
        return self.corpus

    def calc_weighted_pitch_class_values(self):
        for tune in self.corpus:
            onsets = tune.music_data['onset'].to_numpy()
            pitch_class_recalc = tune.music_data['chromatic_pitch_class'].to_numpy()

            eighths = np.arange(int(onsets[0]), int(onsets[-1]) + 1)
            idx = np.searchsorted(onsets, eighths)

            dur_weighted_pitch_class = [pitch_class_recalc[i] for i in idx]
            tune.weighted_music_data[f'dur weighted pitch class'] = dur_weighted_pitch_class

            print('\n' + tune.title + ' weighted feature sequence data:')
            print(tune.weighted_music_data.head())
        return self.corpus

    def save_corpus_feat_seq_data_to_csv(self):
        # TODO: Revisit to allow user-selected outpaths.
        for tune in self.corpus:
            paths_attrs = {
                '/Users/dannydiamond/NUIG/Polifonia/CRE/CRE_secondary_feat_seqs/music_data':
                    tune.music_data,
                '/Users/dannydiamond/NUIG/Polifonia/CRE/CRE_secondary_feat_seqs/weighted_music_data':
                    tune.weighted_music_data,
                '/Users/dannydiamond/NUIG/Polifonia/CRE/CRE_secondary_feat_seqs/music_data_accents':
                    tune.music_data_accents,
                '/Users/dannydiamond/NUIG/Polifonia/CRE/CRE_secondary_feat_seqs/weighted_music_data_accents':
                    tune.weighted_music_data_accents
            }

            for pa in paths_attrs:
                write_to_csv(paths_attrs[pa], pa, tune.title + '_' + pa.split('/')[-1])
                print(f"File sucessfully written: {pa}/{tune.title}_{pa.split('/')[-1]}.csv")

        return None


    # TODO: append output of above to new df, self.keys. self.keys will have tune names as index,
    # TODO: Calculation of mode & certainty has been moved from Music21Corpus to
    #  MusicDataCorpus, but it still needs to be removed from its original location (after testing)
    # TODO: move testing and running to separate files. Can inheritance work across different folders in repo? Check.

# ----------------------------------------------------------------------------------------------------------------------

# Run:

indir = '/Users/dannydiamond/NUIG/Polifonia/MTC/primary_feat_seqs'

# set up corpus:
cre = MusicDataCorpus(indir)
# convert durations to 1/8 notes:
cre.recalc_duration()
cre.recalc_onsets()
# calculate duration-weighted note and velocity seqs:
cre.calculate_duration_weighted_feat_seq_data(['MIDI_note', 'velocity'])
# filter duration-weighted seqs for accented vals only:
cre.filter_duration_weighted_accents()
# find most frequently occurring MIDI note and accented note in 'standard' and duration-weighted sequences:
cre.find_most_freq_notes_in_pitch_seq()
# add the results of the above to a corpus-level dataframe, giving the title and most frequent notes for each tune:
cre.populate_musicdata_roots_df()
# normalize numerical values in above dataframe to one octave:
cre.noramalize_root_values()
# convert numerical root vals to note names:
cre.convert_normalized_roots_to_note_names()
# read music21 roots:
m21_roots = cre.read_music21_roots('/Users/dannydiamond/NUIG/Polifonia/MTC/root_detection/music21_roots.csv')
# combine music21 and MusicDataCorpus root dataframes:
cre.concatenate_music21_and_music_data_roots()
# remove least-accurate freq columns:
# remove_cols_from_dataframe(cre.roots, ['freq acc', 'freq weighted note'])
# calculate simple agreement scores for modal root values:
# cre.calculate_root_result_confidence_scores()
# # find most frequent root:
# cre.find_most_frequent_root_results()
# append expert-assigned root column to MusicDataCorpus.roots:
expert_roots_path = '/Users/dannydiamond/NUIG/Polifonia/MTC/root_detection/expert_assigned_roots_reformatted.csv'
cre.append_expert_assigned_root_data(expert_roots_path)
write_to_csv(cre.roots, '/Users/dannydiamond/NUIG/Polifonia/MTC/root_detection', 'TEST.csv')
# calculate and visualize Pearson correlation matrix for all tunes:
# plt_title = 'Pearson correlation matrix:\n' \
#              'Results of root detection metrics, with mode of results, and expert-assigned root'
# cre.calculate_correlation_matrix('4_selected_algs_mode_exp', plt_title)
# calculate % agreement between mode and expert-assigned root values:
# cre.calculate_percentage_agreement_between_metrics_and_expert_assigned_vals('/Users/dannydiamond/NUIG/Polifonia/CRE/'
#                                                                             'root_detection/aggregated_results/')
# # Save cre.roots table:
write_to_csv(cre.roots_numeric, '/Users/dannydiamond/NUIG/Polifonia/MTC/root_detection', 'mtc_root_detection_NUMS')

# pre-process for Fleiss' Kappa calculation:
# cre.preprocess_root_freq_data()
# run Fleiss' Kappa calculation:
# cre.calculate_fleiss_kappa()
# # calculate confidence:
# cre.calculate_root_result_confidence_scores()
# # print to csv (for user reference):
# cre_outpath = '/Users/dannydiamond/NUIG/Polifonia/CRE/key_detection'
# write_to_csv(cre.roots, cre_outpath, 'MusicDataCorpus_roots.csv')
# # convert root names to numbers:
# cre.convert_root_names_to_root_nums()
# # convert root numbers to MIDI note numbers:
# cre.convert_root_nums_to_midi_note_nums()
# # calculate pitch classes based on roots for all tunes in corpus:
# cre.calc_pitch_class_values()
# # calculate duration-weighted pitch class seqs for each tune, and add to existing 'weighted_music_data' dataframes:
# cre.calc_weighted_pitch_class_values()
# # filter duration-weighted seqs for accented vals only:
# cre.filter_duration_weighted_accents()
# # save weighted accents, unweighted accents, weighted note and unweighted note sequences for every tune to csv
# cre.save_corpus_feat_seq_data_to_csv()
