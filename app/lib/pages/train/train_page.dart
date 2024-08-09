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
  int _refreshTrigger = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadData();
    });
  }

  void _loadData() {
    final sessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    final newCorpusEntriesProvider =
        Provider.of<NewCorpusEntriesProvider>(context, listen: false);

    if (sessionProvider.currentSession != null) {
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
          if (globalSessionProvider.currentSession != null)
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
              ErrorDistributionChart(refreshTrigger: _refreshTrigger),
              newCorpusEntriesProvider.isLoading
                  ? const CircularProgressIndicator()
                  : Text(
                      'New corpus entries: ${newCorpusEntriesProvider.count}'),
              const SizedBox(height: 20),
              Wrap(
                children: [
                  SkillCard(
                    title: '熔炼新语料',
                    description: '使用 1x batch 新语料训练模型',
                    onActivate: () => _smeltNewCorpus(context),
                    isActive: isSmeltingInProgress,
                  ),
                  SkillCard(
                    title: '新老语料对冲',
                    description: '新老语料参半，治疗过拟合',
                    onActivate: () => _smeltNewOld(context),
                    isActive: isSmeltingInProgress,
                  ),
                ],
              )
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
      setState(() {
        _refreshTrigger++; // Trigger a refresh of the ErrorDistributionChart
      });
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

  Future<void> _smeltNewOld(BuildContext context) async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    final newCorpusEntriesProvider =
        Provider.of<NewCorpusEntriesProvider>(context, listen: false);

    setState(() {
      isSmeltingInProgress = true;
    });

    try {
      final result =
          await API.smeltNewOld(globalSessionProvider.currentSession!.id);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content:
                  Text('New old smelting completed. Loss: ${result['loss']}')),
        );
      }
      // Refresh the error distribution and new corpus entries count
      await newCorpusEntriesProvider
          .fetchNewCorpusEntriesCount(globalSessionProvider.currentSession!);
      setState(() {
        _refreshTrigger++; // Trigger a refresh of the ErrorDistributionChart
      });
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error smelting new old: $e')),
        );
      }
    } finally {
      setState(() {
        isSmeltingInProgress = false;
      });
    }
  }
}
