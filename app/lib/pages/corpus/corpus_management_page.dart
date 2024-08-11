import 'package:app/api/api.dart';
import 'package:app/models/corpus.dart';
import 'package:app/pages/learn/learn_chat_page.dart';
import 'package:app/pages/learn/learn_knowledge_page.dart';
import 'package:flutter/material.dart';

class CorpusManagementPage extends StatefulWidget {
  const CorpusManagementPage({super.key});

  @override
  State<CorpusManagementPage> createState() => _CorpusManagementPageState();
}

class _CorpusManagementPageState extends State<CorpusManagementPage> {
  List<Corpus> corpora = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchCorpora();
  }

  Future<void> _fetchCorpora() async {
    setState(() {
      isLoading = true;
    });
    try {
      final corporaData = await API.fetchCorpora();
      setState(() {
        corpora = corporaData.map((data) => Corpus.fromJson(data)).toList();
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error fetching corpora: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Corpus Management'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => _showAddCorpusDialog(context),
          ),
        ],
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : ListView.builder(
              itemCount: corpora.length,
              itemBuilder: (context, index) {
                final corpus = corpora[index];
                return ListTile(
                  title: Text(corpus.name),
                  subtitle: Text(corpus.description),
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => CorpusDetailPage(corpus: corpus),
                    ),
                  ).then((_) =>
                      _fetchCorpora()), // Refresh after returning from detail page
                );
              },
            ),
    );
  }

  void _showAddCorpusDialog(BuildContext context) {
    final nameController = TextEditingController();
    final descriptionController = TextEditingController();

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Add New Corpus'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameController,
                decoration: const InputDecoration(labelText: 'Corpus Name'),
              ),
              TextField(
                controller: descriptionController,
                decoration: const InputDecoration(labelText: 'Description'),
              ),
            ],
          ),
          actions: [
            TextButton(
              child: const Text('Cancel'),
              onPressed: () => Navigator.of(context).pop(),
            ),
            TextButton(
              child: const Text('Add'),
              onPressed: () => _addCorpus(
                  context, nameController.text, descriptionController.text),
            ),
          ],
        );
      },
    );
  }

  void _addCorpus(BuildContext context, String name, String description) async {
    try {
      await API.createCorpus(name: name, description: description);
      if (context.mounted) {
        Navigator.of(context).pop(); // Close the dialog
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Corpus created successfully')),
        );
      }
      // Refresh the page
      _fetchCorpora();
    } catch (e) {
      if (context.mounted) {
        Navigator.of(context).pop(); // Close the dialog
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error creating corpus: $e')),
        );
      }
    }
  }
}

class CorpusDetailPage extends StatefulWidget {
  final Corpus corpus;

  const CorpusDetailPage({super.key, required this.corpus});

  @override
  State createState() => _CorpusDetailPageState();
}

class _CorpusDetailPageState extends State<CorpusDetailPage> {
  late Future<List<dynamic>> _entriesFuture;

  @override
  void initState() {
    super.initState();
    _refreshEntries();
  }

  void _refreshEntries() {
    setState(() {
      _entriesFuture = API.fetchCorpusEntries(corpusId: widget.corpus.id);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.corpus.name),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => _showAddEntryDialog(context),
          ),
        ],
      ),
      body: FutureBuilder(
        future: _entriesFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const CircularProgressIndicator();
          }
          final entries = snapshot.data as List<CorpusEntry>;
          return ListView.separated(
            itemCount: entries.length,
            itemBuilder: (context, index) {
              final entry = entries[index];
              return ListTile(
                title: Text(entry.entryType),
                subtitle: Text(entry.entryType == 'chat'
                    ? '${entry.messages?.length} messages'
                    : '${(entry.content != null && entry.content!.length > 50) ? entry.content?.substring(0, 50) : entry.content}...'),
                trailing: ElevatedButton(
                  child: const Text('Train'),
                  onPressed: () => _trainEntry(context, entry),
                ),
                onTap: () => _editEntry(context, entry),
              );
            },
            separatorBuilder: (context, index) => const Divider(),
          );
        },
      ),
    );
  }

  void _trainEntry(BuildContext context, CorpusEntry entry) async {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return const Center(child: CircularProgressIndicator());
      },
    );

    try {
      final result = await API.trainSingleEntry(entry.id);
      Navigator.of(context).pop(); // 关闭加载对话框

      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text('Training Complete'),
            content: Text(
                'Loss: ${result['loss']}\nTokens trained: ${result['tokens_trained']}'),
            actions: <Widget>[
              TextButton(
                child: const Text('OK'),
                onPressed: () {
                  Navigator.of(context).pop();
                },
              ),
            ],
          );
        },
      );
    } catch (e) {
      Navigator.of(context).pop(); // 关闭加载对话框
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error training entry: $e')),
      );
    }
  }

  void _showAddEntryDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Add New Entry'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                title: const Text('Chat'),
                onTap: () {
                  Navigator.pop(context);
                  _addChatEntry(context);
                },
              ),
              ListTile(
                title: const Text('Knowledge'),
                onTap: () {
                  Navigator.pop(context);
                  _addKnowledgeEntry(context);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  void _addChatEntry(BuildContext context) async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => LearnChatPage(corpus: widget.corpus),
      ),
    );
    if (result == true) {
      _refreshEntries();
    }
  }

  void _addKnowledgeEntry(BuildContext context) async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => LearnKnowledgePage(corpus: widget.corpus),
      ),
    );
    if (result == true) {
      _refreshEntries();
    }
  }

  void _editEntry(BuildContext context, CorpusEntry entry) async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => entry.entryType == 'chat'
            ? LearnChatPage(corpus: widget.corpus, entry: entry)
            : LearnKnowledgePage(corpus: widget.corpus, entry: entry),
      ),
    );
    if (result == true) {
      _refreshEntries();
    }
  }
}
