import 'package:app/api/api.dart';
import 'package:flutter/material.dart';
import 'package:flutter_chat_ui/flutter_chat_ui.dart';
import 'package:flutter_chat_types/flutter_chat_types.dart' as types;

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final List<types.TextMessage> _messages = [];
  final _user = const types.User(id: 'user');
  final _system = const types.User(id: 'system');
  final _assistant = const types.User(id: 'assistant');
  late String currentTimeStamp;

  @override
  void initState() {
    super.initState();
    currentTimeStamp = DateTime.now().millisecondsSinceEpoch.toString();
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
          IconButton(
            icon: const Icon(Icons.edit),
            tooltip: "切换到学习模式",
            onPressed: () {
            },
          ),
          IconButton(
            icon: const Icon(Icons.save),
            tooltip: "保存对话",
            onPressed: () {
            },
          )],
      ),
      body: Chat(
        messages: _messages,
        onSendPressed: _handleSendPressed,
        user: _user,
      ),
    );
  }

  void _addMessage(types.TextMessage message) {
    setState(() {
      _messages.insert(0, message);
    });
  }

  void _handleSendPressed(types.PartialText message) {
    final textMessage = types.TextMessage(
      author: _user,
      createdAt: DateTime.now().millisecondsSinceEpoch,
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      text: message.text,
    );

    _addMessage(textMessage);
    API.chat(messagesToHistory()).then((response) {
      _addMessage(types.TextMessage(
          author: _assistant,
          createdAt: DateTime.now().millisecondsSinceEpoch,
          id: DateTime.now().millisecondsSinceEpoch.toString(),
          text: response));
    });
  }

  List<Map<String, String>> messagesToHistory() {
    return _messages.map((e) {
      if (e.author.id == _system.id) {
        return {'role': 'system', 'content': e.text};
      }
      if (e.author.id == _user.id) {
        return {'role': 'user', 'content': e.text};
      }
      return {'role': 'assistant', 'content': e.text};
    }).toList().reversed.toList();
  }
}
