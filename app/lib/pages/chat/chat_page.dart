import 'package:app/api/api.dart';
import 'package:app/models/corpus.dart' as corpus;
import 'package:app/models/chat.dart';
import 'package:app/providers/global_training_session_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter_chat_ui/flutter_chat_ui.dart';
// ignore: depend_on_referenced_packages
import 'package:flutter_chat_types/flutter_chat_types.dart' as types;
import 'package:provider/provider.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final List<types.TextMessage> _messages = [];
  final ChatSession chatSession = ChatSession([]);

  final _user = const types.User(id: 'user');
  final _system = const types.User(id: 'system');
  final _assistant = const types.User(id: 'assistant');
  late String currentTimeStamp;

  @override
  void initState() {
    super.initState();
    currentTimeStamp = DateTime.now().millisecondsSinceEpoch.toString();

    chatSession
        .addMessage(corpus.Message(role: 'system', content: '你是一个有用的助手。'));
    _messages.add(types.TextMessage(
        author: _system,
        createdAt: DateTime.now().millisecondsSinceEpoch,
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text: '你是一个有用的助手。'));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        actions: [
          // IconButton(
          //   icon: const Icon(Icons.edit),
          //   tooltip: "切换到学习模式",
          //   onPressed: () {},
          // ),
          // IconButton(
          //   icon: const Icon(Icons.save),
          //   tooltip: "保存对话",
          //   onPressed: () {},
          // ),
          IconButton(
            icon: const Icon(Icons.thumb_up),
            tooltip: "标记为正向训练语料",
            onPressed: _createPositiveGradientEntry,
          ),
          IconButton(
            icon: const Icon(Icons.thumb_down),
            tooltip: "标记为反向训练语料",
            onPressed: _createReverseGradientEntry,
          ),
        ],
      ),
      body: Chat(
        messages: _messages,
        onSendPressed: _handleSendPressed,
        user: _user,
      ),
    );
  }

  void _addMessage(types.TextMessage message, String role) {
    setState(() {
      _messages.insert(0, message);
      chatSession.addMessage(corpus.Message(role: role, content: message.text));
    });
  }

  void _handleSendPressed(types.PartialText message) {
    final textMessage = types.TextMessage(
      author: _user,
      createdAt: DateTime.now().millisecondsSinceEpoch,
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      text: message.text,
    );

    _addMessage(textMessage, 'user');
    API.chat(chatSession.toHistory()).then((response) {
      _addMessage(
          types.TextMessage(
              author: _assistant,
              createdAt: DateTime.now().millisecondsSinceEpoch,
              id: DateTime.now().millisecondsSinceEpoch.toString(),
              text: response),
          'assistant');
    });
  }

  Future<void> _createReverseGradientEntry() async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    final currentSession = globalSessionProvider.currentSession;

    if (currentSession == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No active training session')),
      );
      return;
    }

    try {
      final response = await API.createCorpusEntry({
        'corpus_id':
            'reversed_corpus', // You need to provide the correct corpus ID
        'entry_type': 'chat',
        'messages': chatSession.toHistory(),
        'is_reverse_gradient': true,
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Reverse gradient entry created successfully')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error creating reverse gradient entry: $e')),
      );
    }
  }

  Future<void> _createPositiveGradientEntry() async {
    final globalSessionProvider =
        Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
    final currentSession = globalSessionProvider.currentSession;

    if (currentSession == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No active training session')),
      );
      return;
    }

    try {
      final response = await API.createCorpusEntry({
        'corpus_id': 'positive_corpus',
        'entry_type': 'chat',
        'messages': chatSession.toHistory(),
        'is_reverse_gradient': false,
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Positive gradient entry created successfully')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error creating positive gradient entry: $e')),
      );
    }
  }
}
