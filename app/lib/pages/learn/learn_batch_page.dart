import 'package:app/models/chat.dart';
import 'package:app/providers/batch_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class LearnBatchPage extends StatefulWidget {
  const LearnBatchPage({super.key});

  @override
  State<LearnBatchPage> createState() => _LearnBatchPageState();
}

class _LearnBatchPageState extends State<LearnBatchPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        actions: [
          // 清空
          IconButton(
            icon: const Icon(Icons.clear),
            tooltip: "清空",
            onPressed: () {
              Provider.of<BatchProvider>(context, listen: false).clear();
            },
          ),
        ],
      ),
      body: Consumer<BatchProvider>(
        builder: (context, batchProvider, child) {
          return ListView.builder(
            itemCount: batchProvider.corpus.length,
            itemBuilder: (context, index) {
              String title = "";
              dynamic item = batchProvider.corpus[index];
              if (item is ChatSession) {
                title = item.toHistory().toString();
              } else if (item is String) {
                title = item;
              }
              return ListTile(
                title: Text(title),
              );
            },
          );
        },
      ),
      floatingActionButton: TextButton(
        onPressed: () {
          Provider.of<BatchProvider>(context, listen: false).train();
        },
        child: const Text('学'),
      ),
    );
  }
}
