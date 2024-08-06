import 'package:app/pages/train/components/error_distribution_chart.dart';
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
          .fetchNewCorpusEntriesCount(sessionProvider.currentSessionName!);
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
                          'Active session: ${globalSessionProvider.currentSessionName}'),
                      const SizedBox(height: 20),
                      ErrorDistributionChart(
                          sessionName:
                              globalSessionProvider.currentSessionName!),
                      newCorpusEntriesProvider.isLoading
                          ? CircularProgressIndicator()
                          : Text(
                              'New corpus entries: ${newCorpusEntriesProvider.count}'),
                      const SizedBox(height: 20),
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
                            : () => _loadExistingModel(context),
                        child: const Text('Load Selected Model'),
                      ),
                    ],
                  ],
                ),
              ),
            ),
    );
  }

  Future<void> _startNewTrainingSession(BuildContext context) async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    try {
      final result = await API.createNewTrainingSession();
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

  Future<void> _loadExistingModel(BuildContext context) async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    try {
      await API.loadExistingModel(selectedModel!);
      globalSessionProvider.startSession(selectedModel!);
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
