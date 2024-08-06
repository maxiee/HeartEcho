import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:code_text_field/code_text_field.dart';
import 'package:highlight/languages/markdown.dart' as markdown;

class CorpusManagementPage extends StatefulWidget {
  const CorpusManagementPage({Key? key}) : super(key: key);

  @override
  _CorpusManagementPageState createState() => _CorpusManagementPageState();
}

class _CorpusManagementPageState extends State<CorpusManagementPage> {
  List<Corpus> corpora = [];
  Corpus? selectedCorpus;
  List<CorpusEntry> corpusEntries = [];

  @override
  void initState() {
    super.initState();
    fetchCorpora();
  }

  Future<void> fetchCorpora() async {
    final response = await http.get(Uri.parse('http://localhost:1127/corpus'));
    if (response.statusCode == 200) {
      final List<dynamic> data =
          json.decode(json.decode(response.body)['corpora']);
      setState(() {
        corpora = data.map((item) => Corpus.fromJson(item)).toList();
      });
    }
  }

  Future<void> fetchCorpusEntries(String corpusId) async {
    final response = await http.get(
        Uri.parse('http://localhost:1127/corpus/$corpusId/corpus_entries'));
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body)['entries'];
      setState(() {
        corpusEntries = data.map((item) => CorpusEntry.fromJson(item)).toList();
      });
    }
  }

  void _showCreateCorpusDialog() {
    final TextEditingController nameController = TextEditingController();
    final TextEditingController descriptionController = TextEditingController();

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Create New Corpus'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameController,
                decoration: InputDecoration(labelText: 'Corpus Name'),
              ),
              TextField(
                controller: descriptionController,
                decoration: InputDecoration(labelText: 'Description'),
              ),
            ],
          ),
          actions: [
            TextButton(
              child: Text('Cancel'),
              onPressed: () => Navigator.of(context).pop(),
            ),
            TextButton(
              child: Text('Create'),
              onPressed: () async {
                final response = await http.post(
                  Uri.parse('http://localhost:1127/corpus'),
                  headers: {'Content-Type': 'application/json'},
                  body: json.encode({
                    'name': nameController.text,
                    'description': descriptionController.text,
                  }),
                );
                if (response.statusCode == 200) {
                  Navigator.of(context).pop();
                  fetchCorpora();
                }
              },
            ),
          ],
        );
      },
    );
  }

  void _showCreateCorpusEntryDialog(Corpus corpus) {
    final TextEditingController contentController = TextEditingController();
    String selectedType = 'knowledge';

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              title: Text('Create New Corpus Entry'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  DropdownButton<String>(
                    value: selectedType,
                    items: ['knowledge', 'chat'].map((String value) {
                      return DropdownMenuItem<String>(
                        value: value,
                        child: Text(value),
                      );
                    }).toList(),
                    onChanged: (String? newValue) {
                      setState(() {
                        selectedType = newValue!;
                      });
                    },
                  ),
                  selectedType == 'knowledge'
                      ? TextField(
                          controller: contentController,
                          maxLines: 5,
                          decoration: InputDecoration(labelText: 'Content'),
                        )
                      : ElevatedButton(
                          onPressed: () {
                            Navigator.of(context).pop();
                            _navigateToChatEntryPage(corpus);
                          },
                          child: Text('Open Chat Entry Editor'),
                        ),
                ],
              ),
              actions: [
                TextButton(
                  child: Text('Cancel'),
                  onPressed: () => Navigator.of(context).pop(),
                ),
                TextButton(
                  child: Text('Create'),
                  onPressed: () async {
                    if (selectedType == 'knowledge') {
                      final response = await http.post(
                        Uri.parse('http://localhost:1127/corpus_entry'),
                        headers: {'Content-Type': 'application/json'},
                        body: json.encode({
                          'content': contentController.text,
                          'corpus_entry_type': selectedType,
                          'corpus_name': corpus.name,
                        }),
                      );
                      if (response.statusCode == 200) {
                        Navigator.of(context).pop();
                        fetchCorpusEntries(corpus.id);
                      }
                    } else {
                      Navigator.of(context).pop();
                    }
                  },
                ),
              ],
            );
          },
        );
      },
    );
  }

  void _navigateToChatEntryPage(Corpus corpus) {
    Navigator.of(context)
        .push(
          MaterialPageRoute(
            builder: (context) => ChatEntryPage(corpus: corpus),
          ),
        )
        .then((_) => fetchCorpusEntries(corpus.id));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Corpus Management')),
      body: Row(
        children: [
          Expanded(
            flex: 1,
            child: Column(
              children: [
                ListTile(
                  title: Text('Corpora'),
                  trailing: IconButton(
                    icon: Icon(Icons.add),
                    onPressed: _showCreateCorpusDialog,
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: corpora.length,
                    itemBuilder: (context, index) {
                      return ListTile(
                        title: Text(corpora[index].name),
                        subtitle: Text(corpora[index].description),
                        onTap: () {
                          setState(() {
                            selectedCorpus = corpora[index];
                          });
                          fetchCorpusEntries(corpora[index].id);
                        },
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
          VerticalDivider(),
          Expanded(
            flex: 2,
            child: selectedCorpus == null
                ? Center(child: Text('Select a corpus to view entries'))
                : Column(
                    children: [
                      ListTile(
                        title: Text('Entries for ${selectedCorpus!.name}'),
                        trailing: IconButton(
                          icon: Icon(Icons.add),
                          onPressed: () =>
                              _showCreateCorpusEntryDialog(selectedCorpus!),
                        ),
                      ),
                      Expanded(
                        child: ListView.builder(
                          itemCount: corpusEntries.length,
                          itemBuilder: (context, index) {
                            return ListTile(
                              title: Text(corpusEntries[index].content),
                              subtitle:
                                  Text(corpusEntries[index].corpusEntryType),
                              onTap: () {
                                if (corpusEntries[index].corpusEntryType ==
                                    'knowledge') {
                                  _showKnowledgeEntryEditor(
                                      corpusEntries[index]);
                                } else {
                                  _navigateToChatEntryPage(selectedCorpus!);
                                }
                              },
                            );
                          },
                        ),
                      ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }

  void _showKnowledgeEntryEditor(CorpusEntry entry) {
    final codeController = CodeController(
      text: entry.content,
      language: markdown.markdown,
    );

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Edit Knowledge Entry'),
          content: Container(
            width: MediaQuery.of(context).size.width * 0.8,
            height: MediaQuery.of(context).size.height * 0.6,
            child: CodeField(
              controller: codeController,
            ),
          ),
          actions: [
            TextButton(
              child: Text('Cancel'),
              onPressed: () => Navigator.of(context).pop(),
            ),
            TextButton(
              child: Text('Save'),
              onPressed: () async {
                // Implement save logic here
                Navigator.of(context).pop();
                fetchCorpusEntries(selectedCorpus!.id);
              },
            ),
          ],
        );
      },
    );
  }
}

class ChatEntryPage extends StatefulWidget {
  final Corpus corpus;

  const ChatEntryPage({Key? key, required this.corpus}) : super(key: key);

  @override
  _ChatEntryPageState createState() => _ChatEntryPageState();
}

class _ChatEntryPageState extends State<ChatEntryPage> {
  final TextEditingController promptController = TextEditingController();
  final TextEditingController questionController = TextEditingController();
  final CodeController answerController = CodeController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Create Chat Entry')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("System Prompt:"),
            TextField(controller: promptController),
            SizedBox(height: 16),
            Text('Question:'),
            TextField(controller: questionController),
            SizedBox(height: 16),
            Text('Answer:'),
            Expanded(
              child: CodeField(
                controller: answerController,
                expands: true,
                wrap: true,
                textStyle: TextStyle(fontFamily: 'SourceCode'),
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          final response = await http.post(
            Uri.parse('http://localhost:1127/corpus_entry'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({
              'content': json.encode({
                'prompt': promptController.text,
                'question': questionController.text,
                'answer': answerController.text,
              }),
              'corpus_entry_type': 'chat',
              'corpus_name': widget.corpus.name,
            }),
          );
          if (response.statusCode == 200) {
            Navigator.of(context).pop();
          }
        },
        child: Icon(Icons.save),
      ),
    );
  }
}

class Corpus {
  final String id;
  final String name;
  final String description;

  Corpus({required this.id, required this.name, required this.description});

  factory Corpus.fromJson(Map<String, dynamic> json) {
    return Corpus(
      id: json['_id']['\$oid'],
      name: json['name'],
      description: json['description'],
    );
  }
}

class CorpusEntry {
  final String id;
  final String content;
  final String corpusEntryType;

  CorpusEntry({
    required this.id,
    required this.content,
    required this.corpusEntryType,
  });

  factory CorpusEntry.fromJson(Map<String, dynamic> json) {
    return CorpusEntry(
      id: json['_id']['\$oid'],
      content: json['content'],
      corpusEntryType: json['corpus_entry_type'],
    );
  }
}
