import 'package:app/api/api.dart';
import 'package:app/models/corpus.dart';
import 'package:app/pages/learn/learn_chat_page.dart';
import 'package:app/pages/learn/learn_knowledge_page.dart';
import 'package:flutter/material.dart';

class CorpusManagementPage extends StatelessWidget {
  const CorpusManagementPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Corpus Management')),
      body: FutureBuilder(
        future: API
            .fetchCorpora(), // Implement this method to fetch corpora from API
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const CircularProgressIndicator();
          }
          final corporaData = snapshot.data as List<dynamic>;
          final corpora =
              corporaData.map((data) => Corpus.fromJson(data)).toList();
          return ListView.builder(
            itemCount: corpora.length,
            itemBuilder: (context, index) {
              final corpus = corpora[index];
              return ListTile(
                title: Text(corpus.name),
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => CorpusDetailPage(corpus: corpus),
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}

class CorpusDetailPage extends StatelessWidget {
  final Corpus corpus;

  const CorpusDetailPage({super.key, required this.corpus});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(corpus.name)),
      body: FutureBuilder(
        future: API.fetchCorpusEntries(corpus.id), // Implement this method
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const CircularProgressIndicator();
          }
          final entriesData = snapshot.data as List<dynamic>;
          final entries =
              entriesData.map((data) => CorpusEntry.fromJson(data)).toList();
          return ListView.builder(
            itemCount: entries.length,
            itemBuilder: (context, index) {
              final entry = entries[index];
              return ListTile(
                title: Text(entry.entryType),
                subtitle: Text(entry.entryType == 'chat'
                    ? '${entry.messages?.length} messages'
                    : '${entry.content?.substring(0, 50)}...'),
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => entry.entryType == 'chat'
                        ? LearnChatPage(entry: entry)
                        : LearnKnowledgePage(entry: entry),
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
