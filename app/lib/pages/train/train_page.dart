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
  String? selectedModel;
  bool isSmeltingInProgress = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // _loadData();
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

  @override
  Widget build(BuildContext context) {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context);
    final newCorpusEntriesProvider =
        Provider.of<NewCorpusEntriesProvider>(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('炼丹炉'),
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
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                  'Active session: ${globalSessionProvider.currentSession?.name}'),
              const SizedBox(height: 20),
              // ErrorDistributionChart(
              //     sessionName: globalSessionProvider.currentSession!.name),
              // newCorpusEntriesProvider.isLoading
              //     ? const CircularProgressIndicator()
              //     : Text(
              //         'New corpus entries: ${newCorpusEntriesProvider.count}'),
              const SizedBox(height: 20),
              SkillCard(
                title: '熔炼新语料',
                description: '使用 1x batch 新语料训练模型',
                onActivate: () => _smeltNewCorpus(context),
                isActive: isSmeltingInProgress,
              ),
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
