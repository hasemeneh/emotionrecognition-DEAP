using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;
using WebSocketSharp;
using System.Web.Script.Serialization;

namespace EmoRecoClient
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// digunakan untuk membaca file csv yang kemudian  
	/// </summary>
	public partial class MainWindow : Window
	{
		List<string[]> listData = new List<string[]>();
		List<IndividualModel> listPrediksi = new List<IndividualModel>();
		private WebSocket ws;
		private JavaScriptSerializer serializer;

		public MainWindow()
		{
			InitializeComponent();
			// inisialisasi web socket client
			ws = new WebSocket("ws://127.0.0.1:8000/");
			ws.OnMessage += onMessage;
			// inisialisasi json parser
			serializer = new JavaScriptSerializer();
		}
		private void onMessage(object sender, MessageEventArgs e)
		{
			if (e.IsText)
			{
				// mem-parse json
				var data = new JavaScriptSerializer().Deserialize<double[]>(e.Data);
				var individualModel = new IndividualModel()
				{
					Nama = "Subjek " +listPrediksi.Count,
					Valence = Convert.ToInt32(data[0]),
					Arousal = Convert.ToInt32(data[1])
				};
				listPrediksi.Add(individualModel);
			}
			if(listPrediksi.Count == listData.Count)
			{
				// memanggil forrm ListHasilKlasifikasi di main thread
				Application.Current.Dispatcher.Invoke((Action)delegate {
					ListHasilKlasifikasi listHasilKlasifikasi = new ListHasilKlasifikasi(listPrediksi);
					listHasilKlasifikasi.Show();
					this.Close();
				});
			}
		}
		private void Button_Click(object sender, RoutedEventArgs e)
		{
			OpenFileDialog dlg = new OpenFileDialog();
			dlg.DefaultExt = ".csv";
			dlg.Filter = "comma separated value Files (*.csv)|*.csv";


			// Tampilkan OpenFileDialog
			Nullable<bool> result = dlg.ShowDialog();


			// Ambil file dan tampilkan alamat di text box
			if (result == true)
			{
				// Baca Document
				string filename = dlg.FileName;
				namaFile.Text = filename;
				using (var reader = new StreamReader(filename))
				{
					listData.Clear();
					var line = reader.ReadLine();
					while (!reader.EndOfStream)
					{
						line = reader.ReadLine();
						var values = line.Split(',');
						listData.Add(values);
					}
					// tampilkan jumlah data
					jmlData.Text = "" + listData.Count;
				}
			}

		}

		private void Button_Click_1(object sender, RoutedEventArgs e)
		{
			listPrediksi.Clear();
			if (!ws.IsAlive)
			{
				ws.Connect();
			}
			foreach (var item in listData)
			{
				var dataToSend = serializer.Serialize(item);
				ws.Send(dataToSend);
			}
		}
	}
}
