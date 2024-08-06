// ignore_for_file: constant_identifier_names

import 'dart:io';
import 'package:path_provider/path_provider.dart';

import 'package:flutter/material.dart';

enum Mode { Chat, Corpus, Train }

class GlobalProvider extends ChangeNotifier {
  Mode _mode = Mode.Chat;

  late Directory appDirectory;
  late Directory corpusDirectory; // 语料库目录

  GlobalProvider() {
    // 获取应用目录
    getApplicationDocumentsDirectory().then((path) {
      appDirectory = Directory('${path.path}/HomeEcho');
      if (!appDirectory.existsSync()) {
        appDirectory.createSync();
      }
      corpusDirectory = Directory('${appDirectory.path}/corpus');
      if (!corpusDirectory.existsSync()) {
        corpusDirectory.createSync();
      }
    });
  }

  Mode get mode => _mode;

  /// 未分类语料库目录
  Directory get corpusDirInbox {
    final dir = Directory('${corpusDirectory.path}/inbox');
    if (!dir.existsSync()) {
      dir.createSync();
    }
    return dir;
  }

  /// 问答语料库目录
  Directory get corpusDirChat {
    final dir = Directory('${corpusDirectory.path}/chat');
    if (!dir.existsSync()) {
      dir.createSync();
    }
    return dir;
  }

  void changeMode(Mode mode) {
    _mode = mode;
    notifyListeners();
  }
}
