import 'package:app/models/training_session.dart';
import 'package:app/pages/train/components/error_distribution_chart.dart';
import 'package:app/pages/train/components/skill_card.dart';
import 'package:app/providers/new_corpus_entries_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:app/api/api.dart';
import 'package:app/providers/global_training_session_provider.dart';

class TrainPage extends StatelessWidget {
  const TrainPage({super.key});

  @override
  Widget build(BuildContext context) {
    return _TrainPageContent();
  }
}

class _TrainPageContent extends StatefulWidget {
  @override
  _TrainPageContentState createState() => _TrainPageContentState();
}

class _TrainPageContentState extends State<_TrainPageContent> {
  List<String> savedModels = [];
  bool isLoading = true;
  String? selectedModel;
  bool isSmeltingInProgress = false;

  @override
  void initState() {
    super.initState();
    _loadSavedModels();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadData();
    });
  }

  void _loadData() {
    final sessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    final newCorpusEntriesProvider =
        Provider.of<NewCorpusEntriesProvider>(context, listen: false);

    if (sessionProvider.isSessionActive) {
      newCorpusEntriesProvider
          .fetchNewCorpusEntriesCount(sessionProvider.currentSession!);
    }
  }

  Future<void> _loadSavedModels() async {
    setState(() {
      isLoading = true;
    });
    try {
      final models = await API.getSavedModels();
      setState(() {
        savedModels = models;
        isLoading = false;
      });
    } catch (e) {
      if (context.mounted) {
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading models: $e')),
        );
      }
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context);
    final newCorpusEntriesProvider =
        Provider.of<NewCorpusEntriesProvider>(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Training Lab'),
        actions: [
          if (globalSessionProvider.isSessionActive)
            IconButton(
              icon: const Icon(Icons.stop),
              onPressed: () {
                globalSessionProvider.endSession();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Training session ended')),
                );
              },
            ),
        ],
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (globalSessionProvider.isSessionActive) ...[
                      Text(
                          'Active session: ${globalSessionProvider.currentSession?.name}'),
                      const SizedBox(height: 20),
                      ErrorDistributionChart(
                          sessionName:
                              globalSessionProvider.currentSession!.name),
                      newCorpusEntriesProvider.isLoading
                          ? const CircularProgressIndicator()
                          : Text(
                              'New corpus entries: ${newCorpusEntriesProvider.count}'),
                      const SizedBox(height: 20),
                      SkillCard(
                        title: 'New Corpus Smelting',
                        description:
                            'Train the model with a batch of new corpus entries.',
                        onActivate: () => _smeltNewCorpus(context),
                        isActive: isSmeltingInProgress,
                      ),
                    ] else ...[
                      ElevatedButton(
                        onPressed: globalSessionProvider.isSessionActive
                            ? null
                            : () => _startNewTrainingSession(context),
                        child: const Text('Start New Training'),
                      ),
                      const SizedBox(height: 20),
                      const Text('Or load an existing model:'),
                      const SizedBox(height: 10),
                      DropdownButton<String>(
                        value: selectedModel,
                        hint: const Text('Select a model'),
                        items: savedModels.map((String model) {
                          return DropdownMenuItem<String>(
                            value: model,
                            child: Text(model),
                          );
                        }).toList(),
                        onChanged: globalSessionProvider.isSessionActive
                            ? null
                            : (String? newValue) {
                                setState(() {
                                  selectedModel = newValue;
                                });
                              },
                      ),
                      const SizedBox(height: 10),
                      ElevatedButton(
                        onPressed: globalSessionProvider.isSessionActive ||
                                selectedModel == null
                            ? null
                            : () => _loadTrainingSession(context),
                        child: const Text('Load Selected Model'),
                      ),
                    ],
                  ],
                ),
              ),
            ),
    );
  }

  Future<void> _smeltNewCorpus(BuildContext context) async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    final newCorpusEntriesProvider =
        Provider.of<NewCorpusEntriesProvider>(context, listen: false);

    setState(() {
      isSmeltingInProgress = true;
    });

    try {
      final result =
          await API.smeltNewCorpus(globalSessionProvider.currentSession!.id);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text(
                  'New corpus smelting completed. Loss: ${result['loss']}')),
        );
      }
      // Refresh the error distribution and new corpus entries count
      await newCorpusEntriesProvider
          .fetchNewCorpusEntriesCount(globalSessionProvider.currentSession!);
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error smelting new corpus: $e')),
        );
      }
    } finally {
      setState(() {
        isSmeltingInProgress = false;
      });
    }
  }

  Future<void> _startNewTrainingSession(BuildContext context) async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    try {
      final result = await API.createNewTrainingSession(
          'Session_${DateTime.now().millisecondsSinceEpoch}',
          'Qwen/Qwen2-1.5B-Instruct');
      globalSessionProvider.startSession(result);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('New training session started')),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error starting new session: $e')),
        );
      }
    }
  }

  Future<void> _loadTrainingSession(BuildContext context) async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    try {
      final TrainingSession loadedSession = await API
          .loadTrainingSession(globalSessionProvider.currentSession!.id);
      globalSessionProvider.startSession(loadedSession);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Model $selectedModel loaded successfully')),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading model: $e')),
        );
      }
    }
  }
}
