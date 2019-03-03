using System;
using System.Collections.Generic;
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
using System.Windows.Shapes;

namespace EmoRecoClient
{
	/// <summary>
	/// Interaction logic for ListHasilKlasifikasi.xaml
	/// </summary>
	public partial class ListHasilKlasifikasi : Window
	{
		List<IndividualModel> data;
		public ListHasilKlasifikasi(List<IndividualModel> data)
		{
			InitializeComponent();
			this.data = data;
			datagridList.ItemsSource = this.data;
		}

		private void DataGridRow_MouseDoubleClick(object sender, MouseButtonEventArgs e)
		{
			RecoDetail recoDetail = new RecoDetail(data.ElementAt(datagridList.SelectedIndex));
			recoDetail.ShowDialog();
		}

		private void DataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
		{

		}
	}
}
