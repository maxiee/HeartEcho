import 'dart:convert';

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
      appBar: AppBar(
        title: Text(corpus.name),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => _showAddEntryDialog(context),
          ),
        ],
      ),
      body: FutureBuilder(
        future: API.fetchCorpusEntries(
            corpusId: corpus.id), // Implement this method
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
                onTap: () => _editEntry(context, entry),
              );
            },
          );
        },
      ),
    );
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

  void _addChatEntry(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => LearnChatPage(corpus: corpus),
      ),
    );
  }

  void _addKnowledgeEntry(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => LearnKnowledgePage(corpus: corpus),
      ),
    );
  }

  void _editEntry(BuildContext context, CorpusEntry entry) {
    if (entry.entryType == 'chat') {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => LearnChatPage(corpus: corpus, entry: entry),
        ),
      );
    } else {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) =>
              LearnKnowledgePage(corpus: corpus, entry: entry),
        ),
      );
    }
  }
}
